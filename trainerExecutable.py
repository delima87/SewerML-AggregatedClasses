from TrainerHuggingFace import TrainerHF
import argparse
import os 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--n_epochs', type= int, required=True)
    parser.add_argument('--batch_size', type= int, required=True)

    args = parser.parse_args()
    if args.model == 'vit':
        model_checkpoint ='google/vit-base-patch16-224'
    elif args.model == 'resnet50':
        model_checkpoint ='microsoft/resnet-50'
    elif args.model == 'cvt13':
        model_checkpoint ='microsoft/cvt-13'
    elif args.model == 'nvidia':
        model_checkpoint ="nvidia/mit-b2"
    print('model selected', model_checkpoint)
    
    dataset_name = "../../DeepSewerData/DeepSewer/SewerMlData/"
    outName=args.checkpoint
    is_data_local = True
    n_epochs= args.n_epochs
    lr = 2e-5 
    batch_size= args.batch_size
    example_trainer = TrainerHF(model_checkpoint,dataset_name,is_data_local)
    example_trainer.trainModel(outName,n_epochs,lr,batch_size) 
    