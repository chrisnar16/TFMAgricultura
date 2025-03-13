import torch
import torchvision.models as models

def adapt_resnet_channels(model, num_channels):
    # Carga la capa convolucional original
    old_conv_layer = model.conv1

    # Crear una nueva capa convolucional que acepte `num_channels` canales de entrada
    new_conv_layer = torch.nn.Conv2d(
        num_channels,
        old_conv_layer.out_channels,
        kernel_size=old_conv_layer.kernel_size,
        stride=old_conv_layer.stride,
        padding=old_conv_layer.padding,
        bias=old_conv_layer.bias is not None
    )

    # Copia los pesos de los canales existentes en la nueva capa
    with torch.no_grad():
        # Copia los pesos de los primeros min(num_channels, 3) canales
        min_channels = min(num_channels, 3)
        new_conv_layer.weight[:, :min_channels] = old_conv_layer.weight[:, :min_channels]

        # Si hay más canales de entrada que 3, inicializa los pesos adicionales
        if num_channels > 3:
            # Calcula el promedio de los pesos de los primeros 3 canales
            mean_weights = torch.mean(old_conv_layer.weight, dim=1, keepdim=True)
            # Repite este promedio para los canales adicionales
            for i in range(3, num_channels):
                new_conv_layer.weight[:, i] = mean_weights.squeeze(1)

    # Reemplaza la vieja capa convolucional por la nueva
    model.conv1 = new_conv_layer

    return model