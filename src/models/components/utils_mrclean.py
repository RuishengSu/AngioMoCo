import os
from pathlib import Path

from imageio import v3 as iio
from torchmetrics.functional import structural_similarity_index_measure as ssim
# from pytorch_msssim import ssim
from src.models.components import neurite_plot as ne_plot, ndutils


def visualize_series(series, series_fname, dsa, warped_series, dla, series_disp, flow_mags, save_dir, dpi=300):
    ims = []
    dsa_dla_min = min(dsa.min(), dla.min())
    dsa_dla_max = max(dsa.max(), dla.max())
    angio_min = min(series.min(), warped_series.min())
    angio_max = max(series.max(), warped_series.max())
    mask = series[0, :, :]

    for fnum in range(series.shape[0]):
        mse_dsa = dsa[fnum].square().mean()
        mse_dla = dla[fnum].square().mean()
        ssim_dsa = ssim(mask.unsqueeze(0).unsqueeze(0), series[fnum, :, :].unsqueeze(0).unsqueeze(0), data_range=1.0)
        ssim_dla = ssim(mask.unsqueeze(0).unsqueeze(0), warped_series[fnum].unsqueeze(0).unsqueeze(0), data_range=1.0)
        vis = [x.numpy() for x in (mask, series[fnum], warped_series[fnum], series_disp[fnum], dsa[fnum], dla[fnum])]
        im = ne_plot.slices(vis, do_colorbars=True, show=False, return_vis=True, dpi=dpi, grid=(2, 3),
                            imshow_args=[{'vmin': angio_min, 'vmax': angio_max}, {'vmin': angio_min, 'vmax': angio_max},
                                         {'vmin': angio_min, 'vmax': angio_max}, {'vmin': 0, 'vmax': 1},
                                         {'vmin': dsa_dla_min, 'vmax': dsa_dla_max},
                                         {'vmin': dsa_dla_min, 'vmax': dsa_dla_max}],
                            titles=['mask', f'original(frame={fnum:d})', 'registered',
                                    f'disp_field(max={flow_mags[fnum]:.1f} px)',
                                    f'dsa(mse={mse_dsa:.2e}, ssim={ssim_dsa:.2f})',
                                    f'dla(mse={mse_dla:.2e}, ssim={ssim_dla:.2f})'])
        ims.append(im)
    iio.imwrite(os.path.join(save_dir, '{}.gif'.format(series_fname)), ims, duration=1000, loop=0)
    # iio.imwrite(os.path.join(save_dir, '{}.mp4'.format(series_fname)), ims, fps=2)
    # pygifsicle.optimize(os.path.join(save_dir, '{}.gif'.format(series_fname)))  # optional to compress gif


def visualize_pair(mask, target, contrast_fname, vm_warped, disp_field, flow_mag,
                   save_dir, warp_mask=True, dpi=600):
    dsa = target - mask
    dla_vm = (target - vm_warped) if warp_mask else (vm_warped - mask)

    mse_dsa = dsa.square().mean()
    mse_dla = dla_vm.square().mean()
    dsa_dla_min = min(dsa.min(), dla_vm.min())
    dsa_dla_max = max(dsa.max(), dla_vm.max())

    ssim_dsa = ssim(mask.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0), data_range=1.0)
    if warp_mask:
        ssim_dla = ssim(vm_warped.unsqueeze(0).unsqueeze(0), target.unsqueeze(0).unsqueeze(0), data_range=1.0)
    else:
        ssim_dla = ssim(mask.unsqueeze(0).unsqueeze(0), vm_warped.unsqueeze(0).unsqueeze(0), data_range=1.0)

    save_path = os.path.join(save_dir,
                             "{}{}.png".format(contrast_fname, '_warp_mask' if warp_mask else '_warp_contrast'))
    Path(save_path).parent.mkdir(exist_ok=True)
    vis = [x.numpy() for x in (mask, target, vm_warped, disp_field, dsa, dla_vm, dsa-dla_vm)]
    ne_plot.slices(vis, do_colorbars=True, show=False, save_path=save_path, width=20, dpi=dpi, grid=(3, 3),
                   imshow_args=[None, None, None, None, {'vmin': dsa_dla_min, 'vmax': dsa_dla_max},
                                {'vmin': dsa_dla_min, 'vmax': dsa_dla_max}, None],
                   titles=['mask', 'target', 'vm_warped', f'disp_field(max={flow_mag:.1f} px)',
                           f'dsa(mse={mse_dsa:.2e}, ssim={ssim_dsa:.2f})',
                           f'dla(mse={mse_dla:.2e}, ssim={ssim_dla:.2f})',
                           'dsa vs dla_vm'
                           ])
