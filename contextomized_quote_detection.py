from tqdm.auto import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import os
import math
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score

from detection_datasets import(
    create_data_loader,
    make_tensorloader,
)

from models import Encoder, Detection_Model
from util import AverageMeter, set_seed
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from transformers import get_cosine_schedule_with_warmup
from kobert_transformers import get_kobert_model
from kobert_transformers import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--seed", default=0, type=int, help="set seed") 
    parser.add_argument("--split_seed", default=0, type=int, help="seed to split data") 
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")    
    parser.add_argument("--max_len", default=512, type=int, help="max length")     
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers")    
    parser.add_argument("--dimension_size", default=768, type=int, help="dimension size") 
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")    
    parser.add_argument("--classifier_input_size", default=100, type=int, help="input dimension size of classifier") 
    parser.add_argument("--classifier_hidden_size", default=64, type=int, help="hidden size of classifier")     
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="learning rate") 
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay")   
    parser.add_argument("--epochs", default=10, type=int, help="epoch")    
    parser.add_argument("--schedule", default=True, type=bool, help="whether to use the scheduler or not")    
    
    parser.add_argument("--DATA_DIR", default='./data/our_dataset_clean.csv', type=str, help="data to detect contextomized quote")
    parser.add_argument("--MODEL_DIR", default='./model/projection_encoder_best.bin', type=str, help="pretrained QuoteCSE model")
    parser.add_argument("--MODEL_SAVE_DIR", default='./model/contextomized_detection/', type=str, help="fine-tuned QuoteCSE model")
    
    args = parser.parse_args()

    if not os.path.exists(args.MODEL_SAVE_DIR):
        os.makedirs(args.MODEL_SAVE_DIR)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['WANDB_CONSOLE'] = 'off'
    set_seed(args.seed)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    args.backbone_model = get_kobert_model()
    args.tokenizer = get_tokenizer()
    
    df = pd.read_csv(args.DATA_DIR)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.split_seed, stratify=df['label'])
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    ros = RandomOverSampler(random_state=args.seed)
    X_train, y_train = ros.fit_resample(X=df_train.loc[:, ['article_text', 'distorted']].values, y=df_train['label'])
    df_train_ros = pd.DataFrame(X_train, columns=['article_text', 'distorted'])
    df_train_ros['label'] = y_train
    
    loss_func = nn.CrossEntropyLoss(reduction='mean')
    
    encoder = Encoder(args)
    encoder = nn.DataParallel(encoder)
    encoder.load_state_dict(torch.load(args.MODEL_DIR)) 
    encoder = encoder.to(args.device)

    classifier = Detection_Model(4, args)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.to(args.device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()


    if args.schedule:
        total_steps = total_steps = math.ceil(len(df_train_ros) / args.batch_size) * args.epochs
        warmup_steps = math.ceil(len(df_train_ros) / args.batch_size)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = total_steps
        )
        
    
    print('Making Dataloader')
    train_data_loader = create_data_loader(args, df_train_ros, shuffle=True, drop_last=True)
    test_data_loader = create_data_loader(args, df_test, shuffle=False, drop_last=False)

    trainloader =  make_tensorloader(args, encoder, train_data_loader, train=True)
    testloader  = make_tensorloader(args, encoder, test_data_loader)


    loss_data = []
    print('Start Training')
    for epoch in range(args.epochs):
        train_losses = AverageMeter()

        train_loss = []

        tbar = tqdm(trainloader)
        classifier.train()
        for embedding, label in tbar:
            embedding = embedding.to(args.device)
            label = label.to(args.device)

            out = classifier(embedding)
            train_loss = loss_func(out, label)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if args.schedule:
                scheduler.step()

            train_losses.update(train_loss.item(), args.batch_size)
            tbar.set_description("train_loss: {0:.4f}".format(train_losses.avg), refresh=True)

            del out, train_loss

        loss_data.append([epoch, train_losses.avg, 'Train'])

    tbar2 = tqdm(testloader)
    classifier.eval()
    with torch.no_grad():
        predictions = []
        answers = []
        prob_list = []  # AUC용 확률 저장

        for embedding, label in tbar2:
            embedding = embedding.to(args.device)
            label = label.to(args.device)

            out = classifier(embedding)  # [B, 4]
            probs = F.softmax(out, dim=1)  # [B, 4]  → 각 클래스 확률
            preds = torch.argmax(probs, dim=1)  # 예측 라벨

            predictions.extend(preds.cpu().tolist())
            answers.extend(label.cpu().tolist())
            prob_list.extend(probs.cpu().tolist())

            del out, probs, preds

    # list → numpy / list 그대로 써도 sklearn이 처리해 줌
    y_true = answers  # [N], 값은 0/1/2/3
    y_pred = predictions  # [N]
    y_prob = prob_list  # [N, 4] 각 클래스 확률

    # 기본 정확도
    accuracy = accuracy_score(y_true, y_pred)

    # F1 / Precision / Recall: 멀티클래스이므로 average 지정
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # Multi-class AUC (macro, one-vs-rest)
    try:
        auc_macro_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError:
        # 샘플 수가 너무 적거나, 특정 클래스가 아예 안 나온 경우 등에서 터질 수 있음
        auc_macro_ovr = None

    print("===== Evaluation (4-class) =====")
    print("accuracy        :", accuracy)
    print("f1_macro        :", f1_macro)
    print("f1_weighted     :", f1_weighted)
    print("precision_macro :", precision_macro)
    print("recall_macro    :", recall_macro)
    print("auc_macro_ovr   :", auc_macro_ovr)


if __name__ == "__main__":
    main()
