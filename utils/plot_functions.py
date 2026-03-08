import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Union, Any, Dict, Tuple, List, Optional, Sequence

from utils.data_convert_functions import real_imag_to_mag_db_ph_deg

def plot_frequency_responses(
    dataset_to_plot: Sequence[Tuple[torch.Tensor, torch.Tensor, np.ndarray]],
    samples_list: List[Any],
    N_plot_samples: int,
    plot_config: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None
    ) -> None:
    """
    Plots frequency responses for a subset of samples.
    
    Args:
        dataset_to_plot: Sequence of (data_tensor, mask_tensor, freq_array).
        samples_list: List of names/IDs for the samples.
        N_plot_samples: Max number of samples to plot.
        plot_config: Dictionary containing plotting settings.
        save_path: Optional path to save the figure.
    """
    
    # Limit samples to avoid massive figures.
    N_samples_to_plot = min(
        N_plot_samples,
        len(dataset_to_plot),
        plot_config['max_samples_to_plot']
        )
    
    if N_samples_to_plot == 0:
        print("No samples to plot.")
        return

    # 4 plots per sample: 2 columns, 2 rows.
    fig, axs = plt.subplots(
        nrows=2*N_plot_samples,
        ncols=2,
        figsize=(plot_config['fig_width'],
                 2*N_plot_samples * plot_config['fig_height_per_row'])
        )
    
    axs = np.array(axs).flatten()

    for k in range(N_plot_samples):
        try:
            data_tensor, masks_tensor, freq_tensor = dataset_to_plot[k]
        except IndexError:
            break
            
        # Detach from GPU and convert to numpy.
        data = data_tensor.detach().cpu().numpy()
        masks = masks_tensor.detach().cpu().numpy()
        freq = freq_tensor.detach().cpu().numpy()
        
        data_mag_db, data_ph_deg = real_imag_to_mag_db_ph_deg(
            real=data[0,:],
            imag=data[1,:]
            )
        
        x_samples = np.arange(freq.shape[0])
            
        data_map = {
            'freq': freq,
            'samples': x_samples,
            'mag_db': data_mag_db,
            'ph_deg': data_ph_deg,
            'real': data[0,:],
            'imag': data[1,:]
        }

        # Plot 4 subplots for a sample.
        for idx, cfg in enumerate(plot_config['plots']):
            ax = axs[4*k + idx]
            
            # 1. Plot Main Data.
            x_data = data_map[cfg['arg_key']]
            y_data = data_map[cfg['data_key']]
            
            ax.plot(x_data, y_data, '.', markersize=plot_config['markersize_data'], 
                    linestyle='-', alpha=0.7)

            # 2. Plot Masks.
            has_mask = False
            for m_idx, m_cfg in enumerate(plot_config['masks']):
                indices = np.where(masks[m_idx] == 1)[0]
                
                if len(indices) > 0:
                    has_mask = True
                    ax.plot(x_data[indices], y_data[indices], 
                            marker=m_cfg['marker'], markersize=plot_config['markersize_mask'], 
                            linestyle='', color=m_cfg['color'], 
                            label=m_cfg['label'])

            # 3. Styling.
            if idx == 0:
                # Only set title on the first plot of the sample group.
                ax.set_title(f"Item {k}: {samples_list[k]}", fontsize=plot_config['fontsize'], fontweight='bold')
            
            ax.set_xscale(cfg['xscale'])
            ax.set_ylabel(cfg['ylabel'], fontsize=plot_config['fontsize'])
            ax.set_xlabel(cfg['xlabel'], fontsize=plot_config['fontsize'])
            ax.grid(True, alpha=plot_config['grid_alpha'], axis='both', linestyle='--')
            
            # Only show legend if masks exist.
            if has_mask:
                ax.legend(fontsize=plot_config['fontsize_legend'], loc='best', framealpha=0.8)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)