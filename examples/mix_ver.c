#include "darknet.h"
#include "image.h"

#include <sys/time.h>
#include <assert.h>

void run_mixver(char *cfgfile, char *weightfile, char* filename)
{
   	network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

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