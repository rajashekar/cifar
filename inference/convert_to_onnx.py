import torch

from cifer_model import Net


def main():
    # get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    pytorch_model = Net()
    pytorch_model.load_state_dict(torch.load("pretrained/model_cifar.pt", map_location=device))
    # put in eval mode
    pytorch_model.eval()
    # define the input size
    input_size = (3, 32, 32)
    # generate dummy data
    dummy_input = torch.rand(1, *input_size).type(torch.FloatTensor).to(device=device)
    # generate onnx file
    torch.onnx.export(pytorch_model, dummy_input, "pretrained/onnx_model.onnx", verbose=True)


if __name__ == "__main__":
    main()
