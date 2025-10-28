from types import MethodType
from functools import partial

import torch
from comfy.model_base import BaseModel


def apply_model_with_shifted_timestep(
        self: BaseModel,
        x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={},
        shifted_timestep: int = None,
        **kwargs,
):
    device = x.device
    sigma = t.to(device=device, dtype=x.dtype)
    xc = self.model_sampling.calculate_input(sigma, x)
    if c_concat is not None:
        c_concat = c_concat.to(device=device, dtype=self.manual_cast_dtype or self.get_dtype())
        xc = torch.cat((xc, c_concat), dim=1)

    context = c_crossattn
    dtype = self.get_dtype()

    if self.manual_cast_dtype is not None:
        dtype = self.manual_cast_dtype

    xc = xc.to(dtype)
    if shifted_timestep is None:
        t = self.model_sampling.timestep(t).float().to(device)
    else:
        num_train_timesteps = len(self.model_sampling.log_sigmas)
        t = (self.model_sampling.timestep(t) * (shifted_timestep / num_train_timesteps)).long().to(device)

    if context is not None:
        if isinstance(context, (list, tuple)):
            context = type(context)(c.to(device=device, dtype=dtype) for c in context)
        else:
            context = context.to(device=device, dtype=dtype)
    extra_conds = {}
    for k, extra in kwargs.items():
        if torch.is_tensor(extra):
            if extra.dtype in (torch.int64, torch.int32):
                extra = extra.to(device)
            else:
                extra = extra.to(device=device, dtype=dtype)
        extra_conds[k] = extra

    model_output = self.diffusion_model(xc, t, context=context, control=control,
                                        transformer_options=transformer_options, **extra_conds).float()

    if shifted_timestep is None:
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    denoised_sigma = self.model_sampling.sigma(t).to(device)
    denoised_sigma = denoised_sigma.view(denoised_sigma.shape[:1] + (1,) * (x.ndim - 1))
    x = xc * ((denoised_sigma ** 2 + self.model_sampling.sigma_data ** 2) ** 0.5)
    return self.model_sampling.calculate_denoised(denoised_sigma, model_output, x)
    


class TimestepShiftModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "shifted_timestep": ("INT", {"default": 250, "min": 1, "max": 1000}),
                    }
               }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "test"
    FUNCTION = "shift_model_timestep"

    def shift_model_timestep(self, model, shifted_timestep):
        new = model.clone()  # wrapper clone; inner UNet is shared
        if not getattr(new, "_tsm_wrapped", False):
            new.apply_model = MethodType(
                partial(apply_model_with_shifted_timestep, shifted_timestep=shifted_timestep),
                new,
            )
            new._tsm_wrapped = True
        return (new, )


NODE_CLASS_MAPPINGS = {
    "Timestep Shift Model": TimestepShiftModel,
}