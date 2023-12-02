from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np


class PbModel:
    def __init__(self, input_saved_model_file_path: str = None, classes: Tuple = None):
        self.model = tf.keras.models.load_model(input_saved_model_file_path)
        self.grad_model = self.__prepare_grad_model()
        self.model_input_shape = self.model.inputs[0].shape
        self.model_input_dtype = self.model.inputs[0].dtype
        self.model_output_count_shape = self.model.outputs[0].shape
        self.model_output_count_dtype = self.model.outputs[0].dtype
        self.model_output_cam_shape = self.model.outputs[1].shape
        self.model_output_cam_dtype = self.model.outputs[1].dtype
        self.classes = classes

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], Tuple]:
        resized_image_array, resize_image_shape_list, input_image_shape_list = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:3])
        output_count_tensor, output_cam_tensor, output_grad_cam_tensor_list = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(output_count_tensor, output_cam_tensor, output_grad_cam_tensor_list, resize_image_shape_list, input_image_shape_list)
        return output, (output_count_tensor, output_cam_tensor, output_grad_cam_tensor_list)

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> np.ndarray:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        output_count_tensor = tf.zeros((resize_input_tensor.shape[0], self.model_output_count_shape[-1]),
                                       self.model_output_count_dtype).numpy()
        output_cam_tensor = tf.zeros((resize_input_tensor.shape[0], self.model_output_cam_shape[1], self.model_output_cam_shape[2], self.model_output_cam_shape[-1]),
                                     self.model_output_count_dtype).numpy()
        output_grad_cam_tensor_list = [tf.zeros((resize_input_tensor.shape[0], output.shape[1], output.shape[2], self.model_output_count_shape[-1]),
                                                self.model_output_count_dtype).numpy() for output in self.grad_model.outputs[:-2]]

        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_pad = tf.zeros(((batch_size, ) + self.model_input_shape[1:]), self.model_input_dtype).numpy()
            batch_pad[:batch.shape[0], :, :, :] = batch
            raw_count_pred, raw_cam_pred = self.model(batch_pad)
            output_count_tensor[index:index + batch.shape[0], :] = raw_count_pred[:batch.shape[0]]
            output_cam_tensor[index: index + batch.shape[0], :, :, :] = raw_cam_pred[:batch.shape[0]]
        for image_index, resize_input_image in enumerate(resize_input_tensor):
            for pred_index in range(self.model_output_count_shape[-1]):
                raw_grad_list = self.__make_gradcam_heatmap(resize_input_image, pred_index)
                for grad_index, raw_grad in enumerate(raw_grad_list):
                    output_grad_cam_tensor_list[grad_index][image_index, :, :, pred_index] = raw_grad
        return output_count_tensor, output_cam_tensor, output_grad_cam_tensor_list

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list, resize_image_shape_list, input_image_shape_list = [], [], []
        for input_image in input_image_list:
            resized_image, resize_image_shape, input_image_shape = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
            resize_image_shape_list.append(resize_image_shape)
            input_image_shape_list.append(input_image_shape)
        return np.stack(resized_image_list), resize_image_shape_list, input_image_shape_list

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image, resize_image.shape, input_image.shape

    def __output_parse(self, output_count_tensor: np.ndarray, output_cam_tensor: np.ndarray, output_grad_cam_tensor_list: np.ndarray, resize_image_shape_list, input_image_shape_list) -> List[Dict]:
        output_dict_list = []
        for image_index in range(output_count_tensor.shape[0]):
            resize_cam_canvas = np.zeros((input_image_shape_list[image_index][0], input_image_shape_list[image_index][1], output_count_tensor.shape[-1]), dtype=np.float32)
            for pred_index in range(output_count_tensor.shape[-1]):
                resize_cam_canvas[:, :, pred_index] = self.__make_count_heatmap(output_cam_tensor[image_index, :, :, pred_index],
                                                                                resize_image_shape_list[image_index][0], resize_image_shape_list[image_index][1],
                                                                                input_image_shape_list[image_index][0],
                                                                                input_image_shape_list[image_index][1], output_count_tensor[image_index, pred_index])
            resize_grad_cam_canvas_list = []
            for grad_index in range(len(output_grad_cam_tensor_list)):
                resize_grad_cam_canvas =   np.zeros((input_image_shape_list[image_index][0], input_image_shape_list[image_index][1], output_count_tensor.shape[-1]), dtype=np.float32)
                for pred_index in range(output_cam_tensor.shape[-1]):
                    resize_grad_cam_canvas[:, :, pred_index] = self.__make_count_heatmap(output_grad_cam_tensor_list[grad_index][image_index, :, :, pred_index],
                                                                                         resize_image_shape_list[image_index][0], resize_image_shape_list[image_index][1],
                                                                                         input_image_shape_list[image_index][0],
                                                                                         input_image_shape_list[image_index][1], output_count_tensor[image_index, pred_index])
                resize_grad_cam_canvas_list.append(resize_grad_cam_canvas)


            output_dict = {'score': [round(((1.0 - abs(round(count)-count))/2) / 0.5, 4) for count in output_count_tensor[image_index].tolist()],
                           'count': [round(count) for count in output_count_tensor[image_index].tolist()],
                           'cam': resize_cam_canvas,
                           'grad_cam': resize_grad_cam_canvas_list}
            output_dict_list.append(output_dict)
        return output_dict_list


    def __prepare_grad_model(self, layer_name_list=('Conv1', 'block_1_depthwise', 'block_3_depthwise', 'block_6_depthwise')):
        output_layers = []
        for layer_name in layer_name_list:
            output_layers.append(self.model.get_layer(layer_name).output)
        output_layers.append(self.model.output)
        grad_model = tf.keras.models.Model([self.model.inputs], output_layers)
        return grad_model

    def __make_gradcam_heatmap(self, image_array, pred_index=None):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.grad_model(np.expand_dims(image_array, 0))
            last_conv_layer_outputs = outputs[:-1]
            preds = outputs[-1][0]
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        cam_list = []
        for last_conv_layer_output in last_conv_layer_outputs:
            grads = tape.gradient(class_channel, last_conv_layer_output)

            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            cam_list.append(heatmap.numpy())
        del tape
        return cam_list

    def __make_count_heatmap(self, image_array, crop_height, crop_width, resize_height, resize_width, total_num):
        if np.all(image_array == 0.0):
            return np.zeros((resize_height, resize_width), dtype=np.float32)
        org_input_image = Image.fromarray(image_array).resize((self.model_input_shape[2], self.model_input_shape[1]), resample=Image.BICUBIC)
        crop_image = org_input_image.crop((0, 0, crop_width, crop_height))
        resize_crop_image = np.asarray(crop_image.resize((resize_width, resize_height)))
        resize_crop_image = resize_crop_image - np.min(resize_crop_image)
        resize_crop_image = np.clip(resize_crop_image, 0, np.max(resize_crop_image))
        resize_crop_image = resize_crop_image * (total_num/np.sum(resize_crop_image))
        return resize_crop_image