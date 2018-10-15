#include "neural_net.c"
#include "omp.h"

int main(int argc, char *argv[]) {
    iftImage **mask;
    iftMImage **mimg, **cbands, **norm_img;
    bool debug = false;

    if (argc < 4)
        iftError("training <trainX.txt (X=1,2,3,4,5)> <kernel-bank.txt> <output-parameters.txt>", "main");
    else if (argc > 4)
        debug = iftCompareStrings("--debug", argv[4]);
    /* Read input images and kernel bank */

    iftFileSet *trainSet = iftLoadFileSetFromCSV(argv[1], false);
    mask = (iftImage **) calloc(trainSet->n, sizeof(iftImage *));
    mimg = (iftMImage **) calloc(trainSet->n, sizeof(iftMImage *));
    MKernelBank *Kbank = ReadMKernelBank(argv[2]);

    /* Apply the single-layer NN in all training images */
#pragma omp parallel for
    for (int i = 0; i < trainSet->n; i++) {
        printf("Processing file %s\n", trainSet->files[i]->path);
        iftImage *img = iftReadImageByExt(trainSet->files[i]->path);
        mask[i] = ReadMaskImage(trainSet->files[i]->path);
        if ((img->xsize != 352) || (img->ysize != 240))
            printf("imagem %s ", trainSet->files[i]->path);
        mimg[i] = SingleLayer(img, Kbank);
        iftDestroyImage(&img);
    }

    /* Compute plate parameters and normalize activation values within
       [0,255] */

    NetParameters *nparam = CreateNetParameters(mimg[0]->m);
    ComputeAspectRatioParameters(mask, trainSet->n, nparam);
    RegionOfPlates(mask, trainSet->n, nparam);
    //RemoveActivationsOutOfRegionOfPlates(mimg, trainSet->n, nparam);
    norm_img = NormalizeActivationValues(mimg, trainSet->n, 255, nparam);

    /* Find the best kernel weights */
    FindBestKernelWeights(norm_img, mask, trainSet->n, nparam);

    /* Normalize each image */
    //NormalizeImage(mimg, trainSet->n, 255);

    /* Combine bands, find optimum threshold, and apply it */
    cbands = CombineBands(norm_img, trainSet->n, nparam->weight);
    RemoveActivationsOutOfRegionOfPlates(cbands, trainSet->n, nparam);
    FindBestThreshold(cbands, mask, trainSet->n, nparam, 0);

    WriteNetParameters(nparam, argv[3]);
    iftImage **bin = ApplyThreshold(cbands, trainSet->n, nparam);

    /* Post-process binary images and write results on training set */

    PostProcess(bin, trainSet->n, nparam, debug);
    WriteResults(trainSet, bin, debug);

    float * recall = CompareImages(bin, mask, trainSet->n);
    ComputeStats(recall, trainSet->n);

    /* Free memory */

    for (int i = 0; i < trainSet->n; i++) {
        iftDestroyImage(&mask[i]);
        iftDestroyImage(&bin[i]);
        iftDestroyMImage(&cbands[i]);
        iftDestroyMImage(&mimg[i]);
        iftDestroyMImage(&norm_img[i]);
    }
    iftFree(mask);
    iftFree(mimg);
    iftFree(bin);
    iftFree(cbands);
    iftFree(norm_img);
    iftDestroyFileSet(&trainSet);
    DestroyMKernelBank(&Kbank);
    DestroyNetParameters(&nparam);

    return (0);
}
