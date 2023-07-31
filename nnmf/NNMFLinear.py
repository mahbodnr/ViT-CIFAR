import torch
from .utils import calculate_output_size


class NNMFLinear(torch.nn.Module):
    _epsilon_0: float
    _weights: torch.nn.parameter.Parameter
    _weights_exists: bool = False
    _number_of_neurons: int
    _number_of_input_neurons: int
    _h_initial: torch.Tensor | None = None
    _w_trainable: bool
    _weight_noise_range: list[float]
    _input_size: list[int]
    _output_layer: bool = False
    _number_of_iterations: int
    _local_learning: bool = False

    device: torch.device
    default_dtype: torch.dtype

    _number_of_grad_weight_contributions: float = 0.0

    _layer_id: int = -1

    _skip_gradient_calculation: bool

    _keep_last_grad_scale: bool
    _disable_scale_grade: bool
    _last_grad_scale: torch.nn.parameter.Parameter

    def __init__(
        self,
        number_of_input_neurons: int,
        number_of_neurons: int,
        number_of_iterations: int,
        epsilon_0: float = 1.0,
        weight_noise_range: list[float] = [0.0, 1.0],
        w_trainable: bool = False,
        device: torch.device | None = None,
        default_dtype: torch.dtype | None = None,
        layer_id: int = -1,
        local_learning: bool = False,
        output_layer: bool = False,
        skip_gradient_calculation: bool = False,
        keep_last_grad_scale: bool = False,
        disable_scale_grade: bool = True,
    ) -> None:
        super().__init__()

        assert device is not None
        assert default_dtype is not None
        self.device = device
        self.default_dtype = default_dtype

        self._w_trainable = bool(w_trainable)

        self._number_of_input_neurons = int(number_of_input_neurons)
        self._number_of_neurons = int(number_of_neurons)
        self._epsilon_0 = float(epsilon_0)
        self._number_of_iterations = int(number_of_iterations)
        self._weight_noise_range = weight_noise_range
        self._layer_id = layer_id
        self._local_learning = local_learning
        self._output_layer = output_layer
        self._skip_gradient_calculation = skip_gradient_calculation

        self._keep_last_grad_scale = bool(keep_last_grad_scale)
        self._disable_scale_grade = bool(disable_scale_grade)
        self._last_grad_scale = torch.nn.parameter.Parameter(
            torch.tensor(-1.0, dtype=self.default_dtype),
            requires_grad=True,
        )

        self.set_h_init_to_uniform()

        # ###############################################################
        # Initialize the weights
        # ###############################################################

        assert len(self._weight_noise_range) == 2
        weights = torch.empty(
            (
                int(self._number_of_neurons),
                int(self._number_of_input_neurons),
            ),
            dtype=self.default_dtype,
            device=self.device,
        )

        torch.nn.init.uniform_(
            weights,
            a=float(self._weight_noise_range[0]),
            b=float(self._weight_noise_range[1]),
        )
        self.weights = weights

        self.functional_nnmf_linear = FunctionalNNMFLinear.apply

    @property
    def weights(self) -> torch.Tensor | None:
        if self._weights_exists is False:
            return None
        else:
            return self._weights

    @weights.setter
    def weights(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 2
        temp: torch.Tensor = (
            value.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
        )
        temp /= temp.sum(dim=1, keepdim=True, dtype=self.default_dtype)
        if self._weights_exists is False:
            self._weights = torch.nn.parameter.Parameter(temp, requires_grad=True)
            self._weights_exists = True
        else:
            self._weights.data = temp

    @property
    def h_initial(self) -> torch.Tensor | None:
        return self._h_initial

    @h_initial.setter
    def h_initial(self, value: torch.Tensor):
        assert value is not None
        assert torch.is_tensor(value) is True
        assert value.dim() == 1
        assert value.dtype == self.default_dtype
        self._h_initial = (
            value.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
            .requires_grad_(False)
        )

    def update_pre_care(self):
        if self._weights.grad is not None:
            assert self._number_of_grad_weight_contributions > 0
            self._weights.grad /= self._number_of_grad_weight_contributions
            self._number_of_grad_weight_contributions = 0.0

    def update_after_care(self, threshold_weight: float):
        if self._w_trainable is True:
            self.norm_weights()
            self.threshold_weights(threshold_weight)
            self.norm_weights()

    def after_batch(self, new_state: bool = False):
        if self._keep_last_grad_scale is True:
            self._last_grad_scale.data = self._last_grad_scale.grad
            self._keep_last_grad_scale = new_state

        self._last_grad_scale.grad = torch.zeros_like(self._last_grad_scale.grad)

    def set_h_init_to_uniform(self) -> None:
        assert self._number_of_neurons > 2

        self.h_initial: torch.Tensor = torch.full(
            (self._number_of_neurons,),
            (1.0 / float(self._number_of_neurons)),
            dtype=self.default_dtype,
            device=self.device,
        )

    def norm_weights(self) -> None:
        assert self._weights_exists is True
        temp: torch.Tensor = (
            self._weights.data.detach()
            .clone(memory_format=torch.contiguous_format)
            .type(dtype=self.default_dtype)
            .to(device=self.device)
        )
        temp /= temp.sum(dim=1, keepdim=True, dtype=self.default_dtype)
        self._weights.data = temp

    def threshold_weights(self, threshold: float) -> None:
        assert self._weights_exists is True
        assert threshold >= 0

        torch.clamp(
            self._weights.data,
            min=float(threshold),
            max=None,
            out=self._weights.data,
        )

    ####################################################################
    # Forward                                                          #
    ####################################################################

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        # Are we happy with the input?
        assert input is not None
        assert torch.is_tensor(input) is True
        assert input.dim() == 2
        assert input.dtype == self.default_dtype
        assert input.shape[1] == self._number_of_input_neurons

        # Are we happy with the rest of the network?
        assert self._epsilon_0 is not None
        assert self._h_initial is not None
        assert self._weights_exists is True
        assert self._weights is not None

        input.requires_grad_(True)
        input = input / (input.sum(dim=1, keepdim=True) + 1e-20)

        parameter_list = torch.tensor(
            [
                int(0),  # 0
                int(0),  # 1
                int(self._number_of_iterations),  # 2
                int(self._w_trainable),  # 3
                int(self._skip_gradient_calculation),  # 4
                int(self._keep_last_grad_scale),  # 5
                int(self._disable_scale_grade),  # 6
                int(self._local_learning),  # 7
                int(self._output_layer),  # 8
            ],
            dtype=torch.int64,
        )

        epsilon_0 = torch.tensor(self._epsilon_0)

        h = self.functional_nnmf_linear(
            input,
            epsilon_0,
            self._weights,
            self._h_initial,
            parameter_list,
            self._last_grad_scale,
        )

        self._number_of_grad_weight_contributions += h.shape[0]

        return h


class FunctionalNNMFLinear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        epsilon_0: torch.Tensor,
        weights: torch.Tensor,
        h_initial: torch.Tensor,
        parameter_list: torch.Tensor,
        grad_output_scale: torch.Tensor,
    ) -> torch.Tensor:
        number_of_iterations: int = int(parameter_list[2])

        _epsilon_0 = epsilon_0.item()

        h = torch.tile(
            h_initial.unsqueeze(0),
            dims=[
                int(input.shape[0]),
                1,
            ],
        )

        for _ in range(0, number_of_iterations):
            h_w = h.unsqueeze(-1) * weights.unsqueeze(0)
            h_w = h_w / (h_w.sum(dim=1, keepdim=True) + 1e-20)
            h_w = (h_w * input.unsqueeze(1)).sum(dim=-1)
            if _epsilon_0 > 0:
                h = h + _epsilon_0 * h_w
            else:
                h = h_w
            h = h / (h.sum(dim=1, keepdim=True) + 1e-20)

        ctx.save_for_backward(
            input,
            weights,
            h,
            parameter_list,
            grad_output_scale,
        )

        return h

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        #  -> tuple[
        #     torch.Tensor|None,
        #     torch.Tensor|None,
        #     torch.Tensor|None,
        #     torch.Tensor|None,
        #     torch.Tensor|None,
        #     torch.Tensor|None,
        # ]:
        # ##############################################
        # Get the variables back
        # ##############################################
        (
            input,
            weights,
            output,
            parameter_list,
            last_grad_scale,
        ) = ctx.saved_tensors

        # ##############################################
        # Default output
        # ##############################################
        grad_input = None
        grad_epsilon_0 = None
        grad_weights = None
        grad_h_initial = None
        grad_parameter_list = None

        # ##############################################
        # Parameters
        # ##############################################
        parameter_w_trainable: bool = bool(parameter_list[3])
        parameter_skip_gradient_calculation: bool = bool(parameter_list[4])
        parameter_keep_last_grad_scale: bool = bool(parameter_list[5])
        parameter_disable_scale_grade: bool = bool(parameter_list[6])
        parameter_local_learning: bool = bool(parameter_list[7])
        parameter_output_layer: bool = bool(parameter_list[8])

        # ##############################################
        # Dealing with overall scale of the gradient
        # ##############################################
        if parameter_disable_scale_grade is False:
            if parameter_keep_last_grad_scale is True:
                last_grad_scale = torch.tensor(
                    [torch.abs(grad_output).max(), last_grad_scale]
                ).max()
            grad_output /= last_grad_scale
        grad_output_scale = last_grad_scale.clone()

        # input /= input.sum(dim=1, keepdim=True, dtype=weights.dtype) + 1e-20

        # #################################################
        # User doesn't want us to calculate the gradients
        # #################################################

        if parameter_skip_gradient_calculation is True:
            return (
                grad_input,
                grad_epsilon_0,
                grad_weights,
                grad_h_initial,
                grad_parameter_list,
                grad_output_scale,
            )

        # #################################################
        # Calculate backprop error (grad_input)
        # #################################################

        backprop_r: torch.Tensor = weights.unsqueeze(0) * output.unsqueeze(-1)

        backprop_bigr: torch.Tensor = backprop_r.sum(dim=1)

        backprop_z: torch.Tensor = backprop_r * (
            1.0 / (backprop_bigr.unsqueeze(1) + 1e-20)
        )
        grad_input: torch.Tensor = (backprop_z * grad_output.unsqueeze(-1)).sum(1)
        del backprop_z

        # #################################################
        # Calculate weight gradient (grad_weights)
        # #################################################

        if parameter_w_trainable is False:
            # #################################################
            # We don't train this weight
            # #################################################
            grad_weights = None

        elif (parameter_output_layer is False) and (parameter_local_learning is True):
            # #################################################
            # Local learning
            # #################################################
            grad_weights = (
                -2 * (input - backprop_bigr).unsqueeze(1) * output.unsqueeze(2)
            ).sum(0)

        else:
            # #################################################
            # Backprop
            # #################################################
            backprop_f: torch.Tensor = output.unsqueeze(2) * (
                input / (backprop_bigr**2 + 1e-20)
            ).unsqueeze(1)

            result_omega: torch.Tensor = backprop_bigr.unsqueeze(
                1
            ) * grad_output.unsqueeze(2)
            result_omega -= (backprop_r * grad_output.unsqueeze(-1)).sum(2, keepdim=True)
            result_omega *= backprop_f
            del backprop_f
            grad_weights = result_omega.sum(0)
            del result_omega

        del backprop_bigr
        del backprop_r

        return (
            grad_input,
            grad_epsilon_0,
            grad_weights,
            grad_h_initial,
            grad_parameter_list,
            grad_output_scale,
        )


if __name__ == "__main__":
    # test
    layer = NNMFLinear(
        number_of_input_neurons=10,
        number_of_neurons=5,
        number_of_iterations=10,
        epsilon_0=1.0,
        weight_noise_range=[0.0, 1.0],
        w_trainable=True,
        device=torch.device("cpu"),
        default_dtype=torch.float32,
        layer_id=0,
        local_learning=False,
        output_layer=False,
        skip_gradient_calculation=False,
        keep_last_grad_scale=False,
        disable_scale_grade=True,
    )

    input = torch.rand((100, 10), dtype=torch.float32)
    from optimizer import Madam

    optimizer = Madam(
        params=[
            {
                "params": layer.parameters(),
                "nnmf": True,
                "foreach": False,
            },
        ],
    )
    output = layer(input)
    output.sum().backward()
    layer.update_pre_care()
    optimizer.step()
    layer.update_after_care(0.1)
