#include <values.h>
#include "ift.h"
#include "CNNlayer.c"

#define MAX_THRESHOLD 255
#define FALSE_NEGATIVE_CONST 15

typedef struct mkernelbank { /* kernel bank */
    MKernel **K;         /* a vetor of multiband kernels */
    int nkernels;  /* number of kernels */
} MKernelBank;

typedef struct net_parameters { /* parameters of the system */
    float *weight;    /* weight of each band (kernel) */
    int nkernels;  /* number of kernels */
    float threshold; /* final threshold */
    float *maxactiv;  /* maximum activation for normalization per kernel */
    iftBoundingBox bb; /* region in which the training plates are found */
    float mean_width, mean_height; /* mean width and height of the plates */
} NetParameters;

NetParameters *CreateNetParameters(int nkernels) {
    NetParameters *nparam = (NetParameters *) calloc(1, sizeof(NetParameters));
    nparam->weight = iftAllocFloatArray(nkernels);
    nparam->maxactiv = iftAllocFloatArray(nkernels);
    nparam->nkernels = nkernels;
    nparam->threshold = 0.0;
    nparam->mean_width = nparam->mean_height = 0;
    return (nparam);
}

void DestroyNetParameters(NetParameters **nparam) {
    NetParameters *aux = *nparam;
    if (aux != NULL) {
        if (aux->weight != NULL) iftFree(aux->weight);
        if (aux->maxactiv != NULL) iftFree(aux->maxactiv);
        iftFree(aux);
        *nparam = NULL;
    }
}

NetParameters *ReadNetParameters(char *filename) {
    FILE *fp = fopen(filename, "r");
    int nkernels;

    fscanf(fp, "%d", &nkernels);
    NetParameters *nparam = CreateNetParameters(nkernels);
    fscanf(fp, "%f %d %d %d %d %f %f\n", &nparam->threshold, &nparam->bb.begin.x, &nparam->bb.begin.y,
           &nparam->bb.end.x, &nparam->bb.end.y, &nparam->mean_width, &nparam->mean_height);

    for (int i = 0; i < nkernels; i++)
        fscanf(fp, "%f", &nparam->weight[i]);
    for (int i = 0; i < nkernels; i++)
        fscanf(fp, "%f", &nparam->maxactiv[i]);
    fclose(fp);

    return (nparam);
}

void WriteNetParameters(NetParameters *nparam, char *filename) {
    FILE *fp = fopen(filename, "w");

    fprintf(fp, "%d\n", nparam->nkernels);
    fprintf(fp, "%f %d %d %d %d %f %f\n", nparam->threshold, nparam->bb.begin.x, nparam->bb.begin.y, nparam->bb.end.x,
            nparam->bb.end.y, nparam->mean_width, nparam->mean_height);
    for (int i = 0; i < nparam->nkernels; i++)
        fprintf(fp, "%f ", nparam->weight[i]);
    fprintf(fp, "\n");
    for (int i = 0; i < nparam->nkernels; i++)
        fprintf(fp, "%f ", nparam->maxactiv[i]);
    fprintf(fp, "\n");

    fclose(fp);
}

/* Read multi-band kernel bank */

MKernelBank *ReadMKernelBank(char *filename) {
    FILE *fp = fopen(filename, "r");
    iftAdjRel *A;
    int nbands, xsize, ysize, nkernels;
    MKernelBank *Kbank = (MKernelBank *) calloc(1, sizeof(MKernelBank));

    fscanf(fp, "%d %d %d %d", &nbands, &xsize, &ysize, &nkernels);

    A = iftRectangular(xsize, ysize);
    if (A->n != xsize * ysize)
        iftError("Define kernels with odd dimensions (e.g., 3 x 5)", "ReadMKernelBank");

    Kbank->K = (MKernel **) calloc(nkernels, sizeof(MKernel *));
    Kbank->nkernels = nkernels;

    for (int k = 0; k < nkernels; k++) {
        Kbank->K[k] = CreateMKernel(A, nbands);

        for (int i = 0; i < A->n; i++) { // read the weights
            for (int b = 0; b < Kbank->K[0]->nbands; b++) { // for each band
                fscanf(fp, "%f", &Kbank->K[k]->weight[b].val[i]);
            }
        }
        fscanf(fp, "%f", &Kbank->K[k]->bias);
    }

    fclose(fp);

    return (Kbank);
}

void DestroyMKernelBank(MKernelBank **Kbank) {
    MKernelBank *aux = *Kbank;
    for (int k = 0; k < aux->nkernels; k++)
        DestroyMKernel(&aux->K[k]);
    free(aux);
    *Kbank = NULL;
}

iftMatrix *MKernelBankToMatrix(MKernelBank *Kbank) {
    iftMatrix *m_kernel;

    //get info from first kernel (same for all)
    iftAdjRel *A = Kbank->K[0]->A;
    int nbands = Kbank->K[0]->nbands;

    // sum bias in cols
    m_kernel = iftCreateMatrix(A->n * nbands + 1, Kbank->nkernels);

    for (int k = 0; k < Kbank->nkernels; k++) {
        for (int i = 0; i < A->n; i++) {
            for (int b = 0; b < nbands; b++) {
                iftMatrixElem(m_kernel, b + i*nbands, k) = Kbank->K[k]->weight[b].val[i];
            }
        }
        // add bias in end
        iftMatrixElem(m_kernel, A->n * nbands, k) = Kbank->K[k]->bias;
    }

    return m_kernel;

}

iftMImage *SingleLayer(iftImage *img, MKernelBank *Kbank) {
    iftMImage *out = iftCreateMImage(img->xsize, img->ysize, img->zsize, Kbank->nkernels);

    iftAdjRel *A[2];
    iftMImage *aux[2], *mimg;

    if (iftIsColorImage(img)) {
        mimg = iftImageToMImage(img, YCbCr_CSPACE);
    } else {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    }

    A[0] = iftRectangular(7, 3);
    A[1] = iftRectangular(9, 9);

    for (int k = 0; k < Kbank->nkernels; k++) {

        aux[0] = Convolution(mimg, Kbank->K[k]);

        aux[1] = ReLu(aux[0]); /* activation */
        iftDestroyMImage(&aux[0]);

        aux[0] = MaxPooling(aux[1], A[0]);
        iftDestroyMImage(&aux[1]);

        aux[1] = MinPooling(aux[0], A[1]);
        iftDestroyMImage(&aux[0]);


        for (int p = 0; p < out->n; p++) {
            out->band[k].val[p] = aux[1]->band[0].val[p];
        }

        iftDestroyMImage(&aux[1]);
    }

    for (int i = 0; i < 2; i++)
        iftDestroyAdjRel(&A[i]);

    return (out);
}

iftMImage *SingleLayerMatrix(iftImage *img, MKernelBank *Kbank) {

    iftAdjRel *A[2];
    iftMImage *aux[2], *mimg, *out;
    iftMatrix *ximg, *m_kbank, *m_out;

    if (iftIsColorImage(img)) {
        mimg = iftImageToMImage(img, YCbCr_CSPACE);
    } else {
        mimg = iftImageToMImage(img, GRAY_CSPACE);
    }

    ximg = MImageToMatrix(mimg, Kbank->K[0]->A);
    m_kbank = MKernelBankToMatrix(Kbank);

    A[0] = iftRectangular(7, 3);
    A[1] = iftRectangular(9, 9);

    m_out = ConvolutionByMatrixMult(ximg,m_kbank);
    iftDestroyMatrix(&ximg);
    iftDestroyMatrix(&m_kbank);
    aux[0] = MatrixToMImage(m_out,img->xsize, img->ysize, img->zsize);

    aux[1] = ReLu(aux[0]); /* activation */
    iftDestroyMImage(&aux[0]);

    aux[0] = MaxPooling(aux[1], A[0]);
    iftDestroyMImage(&aux[1]);

    out = MinPooling(aux[0], A[1]);
    iftDestroyMImage(&aux[0]);


    for (int i = 0; i < 2; i++)
        iftDestroyAdjRel(&A[i]);

    return (out);
}

void ComputeAspectRatioParameters(iftImage **mask, int nimages, NetParameters *nparam) {
    nparam->mean_width = 0.0;
    nparam->mean_height = 0.0;

    for (int i = 0; i < nimages; i++) { /* For each image */
        iftVoxel pos;
        iftBoundingBox bb = iftMinBoundingBox(mask[i], &pos);
        float width = (float) (bb.end.x - bb.begin.x);
        float height = (float) (bb.end.y - bb.begin.y);

        nparam->mean_width += width;
        nparam->mean_height += height;
    }
    nparam->mean_width /= nimages;
    nparam->mean_height /= nimages;
}

iftMImage **NormalizeActivationValues(iftMImage **mimg, int nimages, int maxval, NetParameters *nparam) {
    float *maxactiv = nparam->maxactiv;

    for (int i = 0; i < nimages; i++) { /* For each image */
        for (int b = 0; b < mimg[i]->m; b++) { /* For each band */
            for (int p = 0; p < mimg[i]->n; p++) { /* Find the maximum
  						activation value */
                if (mimg[i]->band[b].val[p] > maxactiv[b])
                    maxactiv[b] = mimg[i]->band[b].val[p];
            }
        }
    }

    iftMImage **norm_img = (iftMImage **) calloc(nimages, sizeof(iftMImage *));
    for (int i = 0; i < nimages; i++) { /* For each image */
        norm_img[i] = iftCreateMImage(mimg[i]->xsize,mimg[i]->ysize,mimg[i]->zsize,mimg[i]->m);
        for (int b = 0; b < mimg[i]->m; b++) { /* For each band */
            if (maxactiv[b] > 0.0) {
                for (int p = 0; p < mimg[i]->n; p++) { /* Normalize activation values */
                    norm_img[i]->band[b].val[p] = maxval * mimg[i]->band[b].val[p] /  maxactiv[b];

                }
            }
        }
    }

    return norm_img;
}

float *CompareImages(iftImage *const *bins, iftImage *const *masks, int nimages) {
    float *recall = iftAllocFloatArray(nimages);
    for (int i = 0; i < nimages; i++){
        iftImage *bin = bins[i];
        iftImage *mask = masks[i];
        int true_positive = 0, false_negative = 0;

        //iterate image and count errors
        for (int p = 0; p < bin->n; p++) {
            int ground_truth = mask->val[p];
            int pixel = bin->val[p];

            //calculate error
            if (ground_truth == 255){
                if (pixel == 255)
                    true_positive++;
                else
                    false_negative++;
            }
        }

        //computate the error
        recall[i] = true_positive / (float) (true_positive + false_negative);

    }
    return recall;
}

void ComputeStats(float *recall, long n) {
    iftPrintFloatArray(recall, n);
    int hit =0, miss =0;
    for (int i = 0; i < n; i++){
        if (recall[i] > 0.8)
            hit++;
        else
            miss++;
    }
    printf("Hit: %d Miss: %d\n", hit, miss);

}


int *FindThresholdErrors(iftMImage *const *mimg, iftImage *const *masks, int nimages, int b) {
    int *threshold_errors = iftAllocIntArray(MAX_THRESHOLD);
#pragma omp parallel for
    for (int i = 0; i < nimages; i++){
        iftMImage *images_bank = mimg[i];
        iftImage *mask = masks[i];
        iftImage *image_conv = iftMImageToImage(images_bank,255,b);
        for (int threshold =1; threshold <=MAX_THRESHOLD; threshold++) {
            iftImage *image_bin = iftThreshold(image_conv, threshold, 255, 1);
            int false_positive = 0, false_negative = 0;

            //iterate image and count errors
            for (int p = 0; p < image_bin->n; p++) {
                int ground_truth = mask->val[p];
                int pixel = image_bin->val[p];

                //calculate error
                if (ground_truth == 0 && pixel == 1) {
                    false_positive++;
                }
                else if (ground_truth == 255 && pixel == 0){
                    false_negative++;
                }
            }
            iftDestroyImage(&image_bin);

            //computate the error
            threshold_errors[threshold-1] += false_positive + FALSE_NEGATIVE_CONST * false_negative;
        }
        iftDestroyImage(&image_conv);
    }
    return threshold_errors;
}

void FindBestKernelWeights(iftMImage **mimg, iftImage **masks, int nimages, NetParameters *nparam) {
    printf("%s", "FindBestKernelWeights\n");
    float *w = nparam->weight;
    int bands = mimg[0]->m; // same as nkernels
    float *kernel_errors = iftAllocFloatArray(bands);

    for (int b = 0; b < bands; b++) {
        int *threshold_errors = FindThresholdErrors(mimg, masks, nimages, b);

        // find argmax
        int argmin = iftArgmin((const int *) threshold_errors, MAX_THRESHOLD);
        kernel_errors[b] = threshold_errors[argmin]/100.0f;
        if (threshold_errors != NULL) iftFree(threshold_errors);

        /* TODO IT
         * save thresholds to NetParam and apply it in test as well
        // change original image for best thresold
        iftBand band = iftImageToMImage(iftThreshold(image_conv, argmin+1, 255, 1), GRAY_CSPACE)->band[0];
        images_bank->band[b] = band;
         */
    }

    float sum = iftSumFloatArray(kernel_errors,bands);
    for (int i =0; i < bands; i++){
        w[i] = 1-(kernel_errors[i]/sum);
    }
    if (kernel_errors != NULL) iftFree(kernel_errors);


    iftUnitNorm(w, bands);

}

void RegionOfPlates(iftImage **mask, int nimages, NetParameters *nparam) {
    iftBoundingBox bb;

    bb.begin.x = mask[0]->n;
    bb.begin.y = mask[0]->n;
    bb.end.x = -1;
    bb.end.y = -1;
    for (int i = 0; i < nimages; i++) {
        for (int p = 0; p < mask[i]->n; p++) {
            if (mask[i]->val[p] > 0) {
                iftVoxel u = iftGetVoxelCoord(mask[i], p);
                if (u.x < bb.begin.x) bb.begin.x = u.x;
                if (u.y < bb.begin.y) bb.begin.y = u.y;
                if (u.x > bb.end.x) bb.end.x = u.x;
                if (u.y > bb.end.y) bb.end.y = u.y;
            }
        }
    }
    nparam->bb.begin.x = bb.begin.x;
    nparam->bb.end.x = bb.end.x;
    nparam->bb.begin.y = bb.begin.y;
    nparam->bb.end.y = bb.end.y;
}


void RemoveActivationsOutOfRegionOfPlates(iftMImage **mimg, int nimages, NetParameters *nparam) {
    for (int i = 0; i < nimages; i++) { /* For each image */
        for (int p = 0; p < mimg[i]->n; p++) { /* remove activations out
					      of region of plates */
            iftVoxel u = iftMGetVoxelCoord(mimg[i], p);
            if ((u.x < nparam->bb.begin.x) || (u.y < nparam->bb.begin.y) ||
                (u.x > nparam->bb.end.x) || (u.y > nparam->bb.end.y)) {
                for (int b = 0; b < mimg[i]->m; b++) {
                    mimg[i]->band[b].val[p] = 0;
                }
            }
        }
    }
}

iftMImage **CombineBands(iftMImage **mimg, int nimages, float *weight) {
    iftMImage **cbands = (iftMImage **) calloc(nimages, sizeof(iftMImage *));
    iftAdjRel *A[2];

    A[0] = iftRectangular(7, 3);
    A[1] = iftRectangular(9, 9);

    for (int i = 0; i < nimages; i++) {
        cbands[i] = iftCreateMImage(mimg[i]->xsize, mimg[i]->ysize, mimg[i]->zsize, 1);
        for (int p = 0; p < cbands[i]->n; p++) {
            for (int b = 0; b < mimg[0]->m; b++) {
                cbands[i]->band[0].val[p] += weight[b] * mimg[i]->band[b].val[p];
            }
        }
        iftMImage *aux = MaxPooling(cbands[i], A[0]);
        iftDestroyMImage(&cbands[i]);
        cbands[i] = MinPooling(aux, A[1]);
        iftDestroyMImage(&aux);
    }

    iftDestroyAdjRel(&A[0]);
    iftDestroyAdjRel(&A[1]);

    return (cbands);
}

void FindBestThreshold(iftMImage **mimg, iftImage **masks, int nimages, NetParameters *nparam, int band) {
    printf("%s", "FindBestThreshold\n");

    int *threshold_errors = FindThresholdErrors(mimg, masks, nimages, band);

    // find argmin
    nparam->threshold = iftArgmin((const int *) threshold_errors, MAX_THRESHOLD)+1;
    if (threshold_errors != NULL) iftFree(threshold_errors);

    /*
    // change original image for best thresold
    iftBand band = iftImageToMImage(iftThreshold(image_conv, argmax+1, 255, 1), GRAY_CSPACE)->band[0];
    images_bank->band[b] = band;
     */
}

iftImage **ApplyThreshold(iftMImage **cbands, int nimages, NetParameters *nparam) {
    iftImage **bin = (iftImage **) calloc(nimages, sizeof(iftImage *));

    for (int i = 0; i < nimages; i++) {
        bin[i] = iftCreateImage(cbands[i]->xsize, cbands[i]->ysize, cbands[i]->zsize);
        for (int p = 0; p < bin[i]->n; p++) {
            if (cbands[i]->band[0].val[p] >= nparam->threshold)
                bin[i]->val[p] = 255;
        }
    }

    return (bin);
}

void SelectCompClosestTotheMeanWidthAndHeight(iftImage *label, float mean_width, float mean_height) {
    int Lmax = iftMaximumValue(label);
    iftBoundingBox bb;
    float best_width = label->n, best_height = label->n;
    int closest = -1;
    for (int i = 1; i <= Lmax; i++) {
        iftImage *bin = iftExtractObject(label, i);
        iftVoxel pos;
        bb = iftMinBoundingBox(bin, &pos);
        iftDestroyImage(&bin);
        float width = (bb.end.x - bb.begin.x);
        float height = (bb.end.y - bb.begin.y);
        if (fabs(width - mean_width) + fabs(height - mean_height) <
            fabs(best_width - mean_width) + fabs(best_height - mean_height)) {
            best_width = width;
            best_height = height;
            closest = i;
        }
    }

    for (int p = 0; p < label->n; p++)
        if (label->val[p] != closest)
            label->val[p] = 0;
}


void PostProcess(iftImage **bin, int nimages, NetParameters *nparam, bool debug) {
    iftAdjRel *A = iftCircular(sqrtf(2.0));
    iftAdjRel *rec_big = iftRectangular(17,5);
#pragma omp parallel for
    for (int i = 0; i < nimages; i++) {
        char filename[200];
        iftImage *aux[2];
        iftSet *S = NULL;
        aux[0] = iftAddFrame(bin[i], 15, 0);
        sprintf(filename, "result_%d1.png", i);
        if (debug)
            iftWriteImageByExt(aux[0], filename);
        aux[1] = iftDilate(aux[0], rec_big, NULL);
        sprintf(filename, "result_%d2.png", i);
        if (debug)
            iftWriteImageByExt(aux[1], filename);
        iftDestroyImage(&aux[0]);
        aux[0] = iftDilateBin(aux[1], &S, 5.0);
        sprintf(filename, "result_%d3.png", i);
        if (debug)
            iftWriteImageByExt(aux[0], filename);
        iftDestroyImage(&aux[1]);
        aux[1] = iftErodeBin(aux[0], &S, 5.0);
        sprintf(filename, "result_%d4.png", i);
        if (debug)
            iftWriteImageByExt(aux[1], filename);
        iftDestroyImage(&aux[0]);
        aux[0] = iftRemFrame(aux[1], 15);
        iftDestroyImage(&aux[1]);
        iftDestroyImage(&bin[i]);
        iftDestroySet(&S);
        aux[1] = iftFastLabelComp(aux[0], A);
        iftDestroyImage(&aux[0]);
        SelectCompClosestTotheMeanWidthAndHeight(aux[1], nparam->mean_width, nparam->mean_height);
        iftVoxel pos, u, uo, uf;
        iftBoundingBox bb = iftMinBoundingBox(aux[1], &pos);

        u.z = uo.z = uf.z = 0;

        int xsize = bb.end.x - bb.begin.x;
        int ysize = bb.end.y - bb.begin.y;
        int xcenter = bb.begin.x + xsize / 2;
        int ycenter = bb.begin.y + ysize / 2;

        uo.x = iftMax(2, xcenter - iftRound(nparam->mean_width / 2) - 25);
        uo.y = iftMax(2, ycenter - iftRound(nparam->mean_height / 2) - 25);
        uf.x = iftMin(aux[1]->xsize - 3, xcenter + iftRound(nparam->mean_width / 2) + 25);
        uf.y = iftMin(aux[1]->ysize - 3, ycenter + iftRound(nparam->mean_height / 2) + 25);

        bin[i] = iftCreateImage(aux[1]->xsize, aux[1]->ysize, aux[1]->zsize);
        iftDestroyImage(&aux[1]);
        if (uo.x != 0 || uo.y != 0)
            for (u.y = uo.y; u.y <= uf.y; u.y++)
                for (u.x = uo.x; u.x <= uf.x; u.x++) {
                    int p = iftGetVoxelIndex(bin[i], u);
                    bin[i]->val[p] = 255;
                }
    }

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&rec_big);
}

void WriteResults(iftFileSet *fileSet, iftImage **bin, bool debug) {
    iftColor RGB, YCbCr;
    iftAdjRel *A = iftCircular(1.0), *B = iftCircular(sqrtf(2.0));

    RGB.val[0] = 255;
    RGB.val[1] = 100;
    RGB.val[2] = 0;

    YCbCr = iftRGBtoYCbCr(RGB, 255);

    for (int i = 0; i < fileSet->n; i++) {
        iftImage *img = iftReadImageByExt(fileSet->files[i]->path);
        char filename[200];
        iftSList *list = iftSplitString(fileSet->files[i]->path, "_");
        iftSNode *L = list->tail;
        if (debug)
            sprintf(filename, "result_%d9.png", i);
        else
            sprintf(filename, "results/result_%s", L->elem);
        iftDrawBorders(img, bin[i], A, YCbCr, B);
        iftWriteImageByExt(img, filename);
        iftDestroyImage(&img);
        iftDestroySList(&list);
    }

    iftDestroyAdjRel(&A);
    iftDestroyAdjRel(&B);
}

void NormalizeImage(iftMImage **mimg, int nimages, int maxval) {
    float maxactiv = MINFLOAT;
#pragma omp parallel for
    for (int i = 0; i < nimages; i++) { /* For each image */
        for (int b = 0; b < mimg[i]->m; b++) { /* For each band */
            for (int p = 0; p < mimg[i]->n; p++) { /* Find the maximum
  						activation value */
                if (mimg[i]->band[b].val[p] > maxactiv)
                    maxactiv = mimg[i]->band[b].val[p];
            }
            if (maxactiv > 0.0) {
                for (int p = 0; p < mimg[i]->n; p++) { /* Normalize activation values */
                    mimg[i]->band[b].val[p] = maxval * mimg[i]->band[b].val[p] / maxactiv;

                }
            }
        }
    }
}

iftImage *ReadMaskImage(char *pathname) {
    iftImage *mask = NULL;
    iftSList *list = iftSplitString(pathname, "_");
    iftSNode *L = list->tail;
    char filename[200];
    sprintf(filename, "./imagens/placas/mask_%s", L->elem);
    mask = iftReadImageByExt(filename);
    iftDestroySList(&list);
    return (mask);
}
