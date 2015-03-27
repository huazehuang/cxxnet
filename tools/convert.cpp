#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE

#include <ctime>
#include <string>
#include <cstring>
#include <vector>
#include <nnet/nnet.h>
#include <utils/config.h>
#include <nnet/neural_net-inl.hpp>
#include <nnet/nnet_impl-inl.hpp>
#include <layer/convolution_layer-inl.hpp>
#include <layer/cudnn_convolution_layer-inl.hpp>
#include <layer/fullc_layer-inl.hpp>

#include <caffe/blob.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/vision_layers.hpp>

using namespace std;
namespace cxxnet {
    class CaffeConverter {
    public:
        CaffeConverter(){
            this->net_type = 0;
        }

        ~CaffeConverter(){
            if (net_trainer != NULL) {
                delete net_trainer;
            }
        }

        void convert(int argc, char *argv[]){
            if (argc != 5){
                printf("Usage: <caffe_proto> <caffe_model> <cxx_config> <cxx_model_output>");
                return;
            }

            this->initCaffe(argv[1], argv[2]);
            this->initCXX(argv[3]);
            this->transferNet();
            this->SaveModel(argv[4]);
        }

    private:
        inline void initCaffe(const char *network_params, const char *network_snapshot){
            caffe::Caffe::set_mode(caffe::Caffe::CPU);

            caffe_net.reset(new caffe::Net<float>(network_params, caffe::TEST));
            caffe_net->CopyTrainedLayersFrom(network_snapshot);
        }

        inline void initCXX(const char *configPath){
            utils::ConfigIterator itr(configPath);
            while (itr.Next()) {
                this->SetParam(itr.name(), itr.val());
            }

            net_trainer = (nnet::CXXNetThreadTrainer<cpu>*)this->CreateNet();
            net_trainer->InitModel();
        }

        inline void transferNet(){
            const vector<caffe::shared_ptr<caffe::Layer<float> > >& caffeLayers = caffe_net->layers();
            const vector<string> & layer_names = caffe_net->layer_names();

            for (size_t i = 0; i < layer_names.size(); ++i) {
                if (caffe::InnerProductLayer<float> *caffeLayer = dynamic_cast<caffe::InnerProductLayer<float> *>(caffeLayers[i].get())) {
                    printf("Dumping InnerProductLayer %s\n", layer_names[i].c_str());

                    vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffeLayer->blobs();
                    caffe::Blob<float> &caffeWeight = *blobs[0];
                    caffe::Blob<float> &caffeBias = *blobs[1];

                    mshadow::TensorContainer<mshadow::cpu, 2> weight;
                    weight.Resize(mshadow::Shape2(caffeWeight.num(), caffeWeight.channels()));
                    for(int n = 0; n < caffeWeight.num(); n++){
                        for(int c = 0; c < caffeWeight.channels(); c++){
                            weight[n][c] = caffeWeight.data_at(n, c, 0, 0);
                        }
                    }

                    mshadow::TensorContainer<mshadow::cpu, 2> bias;
                    bias.Resize(mshadow::Shape2(caffeBias.count(), 1));
                    for(int b = 0; b < caffeBias.count(); b++){
                        bias[b] = caffeBias.data_at(b, 0, 0, 0);
                    }

                    net_trainer->SetWeight(weight, layer_names[i].c_str(), "wmat");
                    net_trainer->SetWeight(bias, layer_names[i].c_str(), "bias");

                } else if (caffe::ConvolutionLayer<float> *caffeLayer = dynamic_cast<caffe::ConvolutionLayer<float> *>(caffeLayers[i].get())) {
                    printf("Dumping ConvolutionLayer %s\n", layer_names[i].c_str());

                    vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffeLayer->blobs();
                    caffe::Blob<float> &caffeWeight = *blobs[0];
                    caffe::Blob<float> &caffeBias = *blobs[1];


                    mshadow::TensorContainer<mshadow::cpu, 2> weight;
                    weight.Resize(mshadow::Shape2(caffeWeight.num(),
                            caffeWeight.count() / caffeWeight.num()));

                    for(int n = 0; n < caffeWeight.num(); n++){
                        for(int c = 0; c < caffeWeight.channels(); c++){
                            for(int h = 0; h < caffeWeight.height(); h++){
                                for(int w = 0; w < caffeWeight.width(); w++){
                                    float data;
                                    if(i==0) {
                                        data = caffeWeight.data_at(n, 2 - c, h, w);
                                    }else{
                                        data = caffeWeight.data_at(n, c, h, w);
                                    }

                                    weight[n][(c * caffeWeight.height() +
                                               h) * caffeWeight.width() +
                                               w] = data;
                                } // width
                            } // height
                        } // channel
                    } // num

                    mshadow::TensorContainer<mshadow::cpu, 2> bias;
                    bias.Resize(mshadow::Shape2(caffeBias.count(), 1));
                    for(int b = 0; b < caffeBias.count(); b++){
                        bias[b] = caffeBias.data_at(b, 0, 0, 0);
                    }

                    net_trainer->SetWeight(weight, layer_names[i].c_str(), "wmat");
                    net_trainer->SetWeight(bias, layer_names[i].c_str(), "bias");

                } else {
                    printf("Ignoring layer %s\n", layer_names[i].c_str());
                }
            }
        }

        inline void SaveModel(const char *savePath) {
            FILE *fo  = utils::FopenCheck(savePath, "wb");
            fwrite(&net_type, sizeof(int), 1, fo);
            utils::FileStream fs(fo);
            net_trainer->SaveModel(fs);
            fclose(fo);
            printf("Model saved\n");
        }

        inline void SetParam(const char *name , const char *val) {
            cfg.push_back(std::make_pair(std::string(name), std::string(val)));
        }

        // create a neural net
        inline nnet::INetTrainer* CreateNet(void) {
            nnet::INetTrainer *net = nnet::CreateNet<mshadow::cpu>(net_type);

            for (size_t i = 0; i < cfg.size(); ++ i) {
                net->SetParam(cfg[i].first.c_str(), cfg[i].second.c_str());
            }
            return net;
        }

    private:
        /*! \brief type of net implementation */
        int net_type;
        /*! \brief trainer */
        nnet::CXXNetThreadTrainer<cpu> *net_trainer;
    private:
        /*! \brief all the configurations */
        std::vector<std::pair<std::string, std::string> > cfg;
        caffe::shared_ptr<caffe::Net<float> > caffe_net;
    };
}

int main(int argc, char *argv[]){
    cxxnet::CaffeConverter converter;
    converter.convert(argc, argv);
    return 0;
}