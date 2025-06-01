"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as T # Giữ nguyên T cho các transform gốc
import torchvision.transforms.v2.functional as F_tv # Đổi tên để tránh nhầm với torch.nn.functional

from ...core import register # Giả sử đường dẫn này đúng để import register
# Đảm bảo các import này đúng với cấu trúc project của bạn
from .._misc import (
    BoundingBoxes,
    Image, # Lớp Image tùy chỉnh của bạn
    Mask,
    SanitizeBoundingBoxes, # Lớp SanitizeBoundingBoxes tùy chỉnh của bạn
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


# Đăng ký trực tiếp các transform gốc từ torchvision.transforms.v2
# Compose sẽ tạo instance của chúng và gọi __call__ (hoặc forward) của chúng.
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)
# Các transform gốc khác nếu bạn dùng (ví dụ T.ToTensor, T.ConvertImageDtype, T.ToDtype nếu cần thiết)
# Tuy nhiên, ConvertPILImage của bạn có vẻ đã làm nhiệm vụ tương tự ToTensor + ConvertDtype

# Đăng ký transform tùy chỉnh SanitizeBoundingBoxes của bạn
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)


@register()
class EmptyTransform(T.Transform): # Kế thừa từ T.Transform cho nhất quán
    def __init__(
        self,
    ) -> None:
        super().__init__()

    # Hàm forward của T.Transform sẽ tự động xử lý input tuple/list
    # Nếu bạn muốn hành vi cụ thể là trả về input[0] nếu chỉ có 1 phần tử,
    # bạn có thể giữ lại forward này. Nhưng thường T.Transform.forward sẽ làm điều đó.
    # Để đơn giản, có thể chỉ cần implement _transform nếu EmptyTransform thực sự có logic gì đó.
    # Hiện tại, nó không làm gì cả, nên có thể không cần thiết.
    # Nếu mục đích là "no-op", có thể bỏ qua nó trong danh sách ops của Compose.
    # Hoặc, nếu nó là placeholder:
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt
    
    # Nếu bạn muốn forward tùy chỉnh để trả về input[0] khi chỉ có 1 input:
    def forward(self, *inputs: Any) -> Any:
        # Logic này không hoàn toàn khớp với cách T.Transform.forward xử lý,
        # T.Transform.forward sẽ cố gắng áp dụng _transform cho các phần tử phù hợp.
        # Nếu đây là no-op transform, nó sẽ không làm gì với input.
        # inputs_processed = super().forward(*inputs) # Gọi forward của lớp cha để nó xử lý
        # return inputs_processed 
        # Hoặc nếu bạn muốn trả về chính xác cấu trúc input ban đầu:
        return inputs if len(inputs) > 1 else inputs[0]


@register()
class PadToSize(T.Pad): # Kế thừa từ T.Pad là đúng
    _transformed_types = (
        PIL.Image.Image, # Nên dùng Image của torchvision.tv_tensors.Image nếu có thể
        Image,           # Lớp Image tùy chỉnh của bạn
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size) # w, h
        self.target_size = size # Lưu target_size (w, h)
        # T.Pad khởi tạo với padding, không phải target size. self.padding sẽ được tính trong _get_params.
        super().__init__(padding=0, fill=fill, padding_mode=padding_mode) # Khởi tạo padding tạm thời

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # flat_inputs[0] thường là image
        # F_tv.get_spatial_size trả về (height, width)
        spatial_height, spatial_width = F_tv.get_spatial_size(flat_inputs[0])
        
        # target_size là (width, height)
        padding_width = self.target_size[0] - spatial_width
        padding_height = self.target_size[1] - spatial_height

        # T.Pad mong muốn padding là [left, top, right, bottom]
        # Giả sử pad về phía phải và dưới
        self.calculated_padding = [0, 0, max(0, padding_width), max(0, padding_height)] # Đảm bảo không âm
        return {"padding": self.calculated_padding} # Trả về params cho _transform

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # fill đã được xử lý bởi lớp cha T.Pad qua self._fill
        # Chúng ta chỉ cần dùng padding từ params
        current_padding = params["padding"]
        # Hàm F_tv.pad của torchvision.transforms.v2.functional
        return F_tv.pad(inpt, padding=current_padding, fill=self.fill, padding_mode=self.padding_mode)

    # Bỏ __call__ nếu không có logic đặc biệt sau khi super().forward()
    # Hàm forward của T.Pad sẽ tự động gọi _get_params và _transform
    # def __call__(self, *inputs: Any) -> Any:
    #     outputs = super().forward(*inputs) # super().forward() của T.Pad sẽ hoạt động
    #     # Logic thêm 'padding' vào target[1] nếu là dict
    #     # Cần cẩn thận vì T.Pad.forward có thể trả về một object duy nhất hoặc tuple tùy vào input
    #     processed_outputs = list(outputs) if isinstance(outputs, tuple) else [outputs]
    #     if len(processed_outputs) > 1 and isinstance(processed_outputs[1], dict) and hasattr(self, 'calculated_padding'):
    #         # Đảm bảo target là dict và chúng ta muốn thêm padding vào đó
    #         # Tuy nhiên, cấu trúc `sample[-1]` là dataset trong Compose.forward có thể khác.
    #         # Thông thường, input cho transform là (image, target_dict)
    #         if isinstance(inputs[1], dict): # Giả sử input thứ 2 là target dict
    #             # Tạo một bản sao của target để không thay đổi target gốc nếu nó được dùng ở nơi khác
    #             new_target = inputs[1].copy() if len(inputs) > 1 else {}
    #             new_target["padding"] = torch.tensor(self.calculated_padding)
    #             if len(processed_outputs) > 1:
    #                 processed_outputs[1] = new_target
    #             else: # Trường hợp chỉ có image được transform
    #                 return processed_outputs[0], new_target # Trả về cả hai

    #     return tuple(processed_outputs) if len(processed_outputs) > 1 else processed_outputs[0]
    # Để an toàn và đơn giản, hãy để T.Pad.forward() tự xử lý. Nếu cần sửa target,
    # có thể tốt hơn là tạo một transform riêng chỉ để sửa target sau khi PadToSize.
    # HOẶC, nếu bạn muốn PadToSize trả về (transformed_image, new_target_dict_with_padding):
    def forward(self, *inputs: Any) -> Any:
        # super().forward sẽ áp dụng _transform cho các _transformed_types
        outputs = super().forward(*inputs) # outputs có thể là tuple hoặc 1 item

        # Logic thêm 'padding' vào target nếu có
        # Giả sử inputs là (image, target_dict) hoặc (image, bboxes, labels, ...)
        # và chúng ta muốn thêm 'padding' vào target_dict (nếu input thứ 2 là dict)
        if len(inputs) > 1 and isinstance(inputs[1], dict) and hasattr(self, 'calculated_padding'):
            # outputs có thể là tuple (transformed_img, transformed_target)
            # hoặc transformed_img nếu target không phải là _transformed_types
            
            # Tạo một bản sao của target gốc để thêm padding
            # Vì target gốc có thể không được transform bởi super().forward() nếu nó không phải là tv_tensor
            target_copy = inputs[1].copy() 
            target_copy["padding"] = torch.tensor(self.calculated_padding, device=inputs[0].device if isinstance(inputs[0], torch.Tensor) else 'cpu')

            if isinstance(outputs, tuple): # Nếu super().forward trả về nhiều items
                # Giả sử item đầu là image, item thứ hai là target đã transform (nếu có)
                # Chúng ta muốn trả về (transformed_img, target_copy_with_padding)
                return outputs[0], target_copy
            else: # Nếu super().forward chỉ trả về transformed_img
                return outputs, target_copy
        
        return outputs # Trả về output gốc từ super().forward nếu không có target dict


@register()
class RandomIoUCrop(T.RandomIoUCrop): # Kế thừa từ T.RandomIoUCrop
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1.0, # Sửa giá trị mặc định nếu cần
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2.0, # Sửa giá trị mặc định nếu cần
        sampler_options: Optional[List[float]] = None, # Giữ nguyên
        trials: int = 40,
        p: float = 1.0, # Xác suất áp dụng
    ):
        super().__init__(
            min_scale=min_scale, max_scale=max_scale, 
            min_aspect_ratio=min_aspect_ratio, max_aspect_ratio=max_aspect_ratio, 
            sampler_options=sampler_options, trials=trials
        )
        self.p = p

    # T.RandomIoUCrop đã có logic forward. Chúng ta chỉ cần thêm xác suất p.
    # Cách tốt nhất là dùng T.RandomApply nếu có thể.
    # Nếu muốn ghi đè forward để thêm p:
    def forward(self, *inputs: Any) -> Any:
        if torch.rand(1).item() < self.p:
            return super().forward(*inputs) # Gọi forward của T.RandomIoUCrop
        else:
            return inputs if len(inputs) > 1 else inputs[0]
    # Lưu ý: Ghi đè __call__ như bạn làm cũng được, nhưng forward là chuẩn hơn.


@register()
class ConvertBoxes(T.Transform): # Kế thừa từ T.Transform
    _transformed_types = (BoundingBoxes,) # Chỉ áp dụng cho BoundingBoxes tv_tensor

    def __init__(self, fmt: str = "", normalize: bool = False) -> None:
        super().__init__()
        self.fmt = fmt.lower() # Đảm bảo fmt là chữ thường
        self.normalize = normalize

    # Bỏ phương thức `transform`, chỉ cần `_transform`
    def _transform(self, inpt: BoundingBoxes, params: Dict[str, Any]) -> BoundingBoxes: # Type hint rõ ràng hơn
        # `inpt` ở đây chắc chắn là một tv_tensor BoundingBoxes
        spatial_size = inpt.spatial_size # Lấy spatial_size từ tv_tensor

        out_boxes = inpt # Bắt đầu với input
        if self.fmt and self.fmt != inpt.format.value.lower(): # Chỉ convert nếu fmt khác và có giá trị
            out_boxes = torchvision.ops.box_convert(inpt, in_fmt=inpt.format.value, out_fmt=self.fmt)
            # Sau box_convert, nó là tensor thường, cần tạo lại BoundingBoxes tv_tensor
            out_boxes = BoundingBoxes(out_boxes, format=self.fmt.upper(), spatial_size=spatial_size, canvas_size=inpt.canvas_size)


        if self.normalize:
            # Chuẩn hóa bằng spatial_size (height, width)
            # box / [W, H, W, H]
            normalizer = torch.tensor([spatial_size[1], spatial_size[0], spatial_size[1], spatial_size[0]], 
                                      dtype=out_boxes.dtype, device=out_boxes.device)
            out_boxes_data = out_boxes / normalizer
            out_boxes = BoundingBoxes(out_boxes_data, format=out_boxes.format, spatial_size=spatial_size, canvas_size=inpt.canvas_size)

        return out_boxes


@register()
class ConvertPILImage(T.Transform): # Kế thừa từ T.Transform
    _transformed_types = (PIL.Image.Image,) # Chỉ áp dụng cho PIL Image

    def __init__(self, dtype: str ="float32", scale: bool =True) -> None:
        super().__init__()
        self.target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.scale = scale

    # Bỏ phương thức `transform`, chỉ cần `_transform`
    def _transform(self, inpt: PIL.Image.Image, params: Dict[str, Any]) -> Image: # Trả về Image tv_tensor của bạn
        out_tensor = F_tv.pil_to_tensor(inpt) # Chuyển PIL thành tensor C, H, W uint8
        
        # Chuyển dtype
        out_tensor = F_tv.to_dtype(out_tensor, dtype=self.target_dtype, scale=self.scale) # scale=True sẽ chia cho 255 nếu input là uint8 và target là float
        
        # Bọc bằng lớp Image tùy chỉnh của bạn
        return Image(out_tensor)


@register()
class RandomRotate90(T.Transform):
    _transformed_types = (
        PIL.Image.Image, # Nên chuyển sang Image tv_tensor trước khi dùng transform này
        Image,           # Lớp Image tùy chỉnh của bạn (nên là tv_tensor.Image)
        Video,
        Mask,
        BoundingBoxes,
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"probability p should be between 0 and 1, but got {p}")
        self.p = p # Lưu xác suất

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        if torch.rand(1).item() >= self.p: # Quyết định có áp dụng transform hay không
            return {"apply_transform": False, "angle": 0.0} 
        
        k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 (tương ứng 90, 180, 270 độ)
        angle = float(k * 90)
        return {"apply_transform": True, "angle": angle}

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if not params["apply_transform"]: # Nếu không áp dụng, trả về input gốc
            return inpt
        
        angle = params["angle"]
        # F_tv.rotate xử lý các tv_tensor và cập nhật spatial_size đúng cách
        return F_tv.rotate(inpt, angle=angle, expand=False) # expand=False cho xoay 90 độ

    # Bỏ hàm forward tùy chỉnh. Hàm forward của T.Transform (lớp cha) sẽ tự động:
    # 1. Gọi _get_params để lấy "angle" và "apply_transform".
    # 2. Nếu "apply_transform" là True, nó sẽ gọi _transform với input và params đó.
    # 3. Xử lý nhiều input (tuple/list) và các kiểu dữ liệu khác nhau.
    # def forward(self, *inputs: Any) -> Any:
    #     # ... (logic cũ của bạn với super().forward() đã bị xóa) ...
    #     # Bây giờ, chỉ cần dựa vào forward của T.Transform
    #     pass # Hoặc không cần định nghĩa forward ở đây, nó sẽ kế thừa từ T.Transform