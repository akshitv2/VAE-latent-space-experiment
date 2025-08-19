import torch
from experiments.ModelLoader import load_encoder_decoder
import ipywidgets as widgets
from IPython.display import display, clear_output

def inference(checkpoint_path: str = "./experiments/checkpoints/vae_checkpoint_epoch_10.pt", device: torch.device = None):
    #encoder, decoder = load_encoder_decoder(checkpoint_path)
    latent_dim = 20
    sliders = []
    for i in range(latent_dim):
        slider = widgets.FloatSlider(
            value=0.0,
            min=-3.0,
            max=3.0,
            step=0.1,
            description=f'z{i}:',
            continuous_update=True,
            orientation='horizontal',
            layout=widgets.Layout(width='400px')
        )
        sliders.append(slider)

    slider_box = widgets.VBox(sliders)
    display(slider_box)
    # for s in sliders:
    #     s.observe(plot_value_from_slider, names='value')

if __name__ == "__main__":
    inference()