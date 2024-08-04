import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from mymodels import TransDataset
from torch.optim import lr_scheduler
from PIL import Image

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return (height, width)

def create_dataloaders(df_train, df_valid, df_test, ycol, batch_size):

    sample_image_path = df_train["filepath"].iloc[0]
    img_size = get_image_size(sample_image_path)
    print("H*W: ", img_size)
    
    label_encoder = LabelEncoder()
    df_train[ycol] = label_encoder.fit_transform(df_train[ycol])
    df_valid[ycol] = label_encoder.transform(df_valid[ycol])
    df_test[ycol] = label_encoder.transform(df_test[ycol])
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    train_dataset = TransDataset(df_train, img_size, ycol, transform=train_transform)
    valid_dataset = TransDataset(df_valid, img_size, ycol, transform=valid_transform)
    test_dataset = TransDataset(df_test, img_size, ycol, transform=valid_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, num_epochs=25, lr=1e-5):

    train_losses = []
    valid_losses = []
    train_f1s = []
    valid_f1s = []
    train_aucs = []
    valid_aucs = []
    train_accuracies = []
    valid_accuracies = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    # *** あとで試してみてください ***
    # *** 下にある scheduler.step() のコメントを戻すのも忘れないようにしてください ***
    # warmup = num_epoch // 5
    # def get_lr(epo):
    #     if it < warmup:
    #         return (epo + 1) / warmup
    #     else:
    #         return 1 - (epo - warmup) / (num_epoch - warmup)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # *** 追加 ***
            if (i + 1) % 10 == 0:
                print(f"epoch: {epoch} iter: {i} loss: {loss}")

            #print("outputs: ", outputs[0])
            #print("outputs shape: ", outputs.shape)
            #print("outputs shape squeeze: ", outputs.squeeze())
            #print("loss: ", loss)
            loss.backward()

            # *** 追加 ***
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        #epoch_f1 = f1_score(all_labels, [1 if x >= 0.5 else 0 for x in all_preds])
        epoch_f1 = f1_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])
        #epoch_auc = roc_auc_score(all_labels, all_preds)
        #epoch_acc = accuracy_score(all_labels, [1 if x >= 0.5 else 0 for x in all_preds])
        epoch_acc = accuracy_score(all_labels, [1 if x >= 0 else 0 for x in all_preds])

        train_losses.append(epoch_loss)
        train_f1s.append(epoch_f1)
        #train_aucs.append(epoch_auc)
        train_accuracies.append(epoch_acc)

        #print(f'Epoch {epoch}/{num_epochs - 1} | Loss: {epoch_loss:.4f} | F1: {epoch_f1:.4f} | AUC: {epoch_auc:.4f}')
        
        model.eval() # Validation だから。
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device) # data -> GPU
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels) # バッチのデフォルトは平均値がかえってくる
                # *** 追加 ***
                if (i + 1) % 10 == 0:
                    print(f"[VAL] epoch: {epoch} iter: {i} loss: {loss}")
                
                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        print(f"[VAL] val_loss: {val_loss}")
        # *** よくみたらもとのコードでここはOKでした ***
        val_loss /= len(valid_loader.dataset)
        print(f"[VAL] val_loss(div): {val_loss}")

        #val_f1 = f1_score(val_labels, [1 if x >= 0.5 else 0 for x in val_preds])
        val_f1 = f1_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])
        #print("val_labels: ", val_labels)
        #val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, [1 if x >= 0 else 0 for x in val_preds])
        #val_acc = accuracy_score(val_labels, [1 if x >= 0.5 else 0 for x in val_preds])

        valid_losses.append(val_loss)
        valid_f1s.append(val_f1)
        #valid_aucs.append(val_auc)
        valid_accuracies.append(val_acc)

        #print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}')

        # *** あとで試してみてください ***
        # scheduler.step()
        
    epochs = range(num_epochs)
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, valid_f1s, label='Valid F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    
    #plt.subplot(2, 2, 4)
    #plt.plot(epochs, train_aucs, label='Train AUC')
    #plt.plot(epochs, valid_aucs, label='Valid AUC')
    #plt.xlabel('Epoch')
    #plt.ylabel('AUC')
    #plt.title('AUC')
    #plt.legend()
    
    plt.tight_layout()
    plt.show()
