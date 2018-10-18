#include <getopt.h>
#include "ift.h"
#include "neural_net.c"

int main(int argc, char *argv[]) {
    iftImage **mask;
    iftMImage **mimg, **cbands, **norm_img;
    NetParameters *nparam;
    static int debug, matrix=1;

    if (argc < 4)
        iftError("testing <testX.txt (X=1,2,3,4,5)> <kernel-bank.txt> <input-parameters.txt> [--debug] [--matrix] [--interactive]", "main");
    else if (argc > 4) {
        static struct option long_options[] =
                {
                        {"debug", no_argument,       &debug, 1},
                        {"interactive",   no_argument,    &matrix, 0},
                        {"matrix",   no_argument,    &matrix, 1},
                };
        /* getopt_long stores the option index here. */
        int option_index = 4;

        while (getopt_long_only(argc, argv, "",long_options, &option_index) != -1);
    }
    /* Read input images and kernel bank */

    iftFileSet *testSet = iftLoadFileSetFromCSV(argv[1], false);
    mask = (iftImage **) calloc(testSet->n, sizeof(iftImage *));
    mimg = (iftMImage **) calloc(testSet->n, sizeof(iftMImage *));
    MKernelBank *Kbank = ReadMKernelBank(argv[2]);
    nparam = ReadNetParameters(argv[3]);

    /* Apply NN in all test images */
#pragma omp parallel for
    for (int i = 0; i < testSet->n; i++) {
        printf("Processing file %s\n", testSet->files[i]->path);
        iftImage *img = iftReadImageByExt(testSet->files[i]->path);
        mask[i] = ReadMaskImage(testSet->files[i]->path);
        if (matrix)
            mimg[i] = SingleLayerMatrix(img, Kbank);
        else
            mimg[i] = SingleLayer(img, Kbank);
        iftDestroyImage(&img);
    }

    /* Normalize activation values within [0,255] */

    //RemoveActivationsOutOfRegionOfPlates(mimg, testSet->n, nparam);
    norm_img = NormalizeActivationValues(mimg, testSet->n, 255, nparam);

    /* Combine bands */

    cbands = CombineBands(norm_img, testSet->n, nparam->weight);
    RemoveActivationsOutOfRegionOfPlates(cbands, testSet->n, nparam);

    iftImage **bin = ApplyThreshold(cbands, testSet->n, nparam);

    /* Post-process binary images and write results on training set */

    PostProcess(bin, testSet->n, nparam, debug);
    WriteResults(testSet, bin, debug);

    float * recall = CompareImages(bin, mask, testSet->n);
    ComputeStats(recall, testSet->n);

    /* Free memory */

    for (int i = 0; i < testSet->n; i++) {
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
    iftDestroyFileSet(&testSet);
    DestroyMKernelBank(&Kbank);
    DestroyNetParameters(&nparam);


    return (0);
}
