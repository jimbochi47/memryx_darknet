#include "darknet.h"
#include "image.h"

#include <sys/time.h>
#include <assert.h>

char* l_type[] = {
    "CONVOLUTIONAL",
    "DECONVOLUTIONAL",
    "CONNECTED",
    "MAXPOOL",
    "SOFTMAX",
    "DETECTION",
    "DROPOUT",
    "CROP",
    "ROUTE",
    "COST",
    "NORMALIZATION",
    "AVGPOOL",
    "LOCAL",
    "SHORTCUT",
    "ACTIVE",
    "RNN",
    "GRU",
    "LSTM",
    "CRNN",
    "BATCHNORM",
    "NETWORK",
    "XNOR",
    "REGION",
    "YOLO",
    "ISEG",
    "REORG",
    "UPSAMPLE",
    "LOGXENT",
    "L2NORM",
    "BLANK"
};

void set_weights(network* net, char* weightfile) {

    printf("Loading weigts files: %s \n", weightfile);
    FILE* input_file = fopen(weightfile, "r");
    if (!input_file) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < net->n; ++i){
        layer l = net->layers[i];
        printf("layer type is %s\n", l_type[l.type]);
        if(l.type == CONVOLUTIONAL) {
            int num = l.c/l.groups*l.n*l.size*l.size;
            float* weight_array = calloc((num+l.n), sizeof(float));
            char line_buffer[20];
            int i = 0;
            while(fgets(line_buffer, sizeof(line_buffer), input_file) != NULL) {
                if(i >= num)
                    break;
                line_buffer[strcspn(line_buffer, "\n")] = 0;
                // printf("jimbo inside: %s \n", line_buffer);
                weight_array[i] = atof(line_buffer);
                i++;
            }

            for(i = 0; i < l.n; i++) {
                l.biases[i] = weight_array[i];
                printf("convolutional biases: %f \n", l.biases[i]);
            }
            for(i = 0; i < num; i++) {
                l.weights[i] = weight_array[i+l.n];
                printf("convolutional weights: %f \n", l.weights[i]);
            }
        }
    }
}

void run_mixver(char *cfgfile, char *weightfile, char* filename)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    set_weights(net, weightfile);

    image img_data_in;
    if(filename) {
        img_data_in = load_array_to_img(filename, net->w ,net->h ,net->c);
    }
    else {
        img_data_in = make_random_image(net->w ,net->h ,net->c);
        save_random_image_as_array(img_data_in);
    }
    float *data_in = img_data_in.data;

    float *predictions = network_predict(net, data_in);
    if(net->hierarchy) hierarchy_predictions(predictions ,net->outputs ,net->hierarchy ,1 ,1);

    int i;
    for(i = 0; i < net->outputs; i++)
        printf("Predictions results: %lf\n", predictions[i]);

    free(data_in);
}
