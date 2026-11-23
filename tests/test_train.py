import torch
from roadsign.model import RoadSignNet
from roadsign.train import train_one_epoch, evaluate


def test_train_and_eval_one_epoch():
    model = RoadSignNet(num_classes=10)
    device = torch.device("cpu")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch
    imgs = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    loader = [(imgs, labels)]

    loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
    assert 0 <= acc <= 1
    assert loss > 0

    val_loss, val_acc = evaluate(model, loader, criterion, device)
    assert 0 <= val_acc <= 1
    assert val_loss > 0
