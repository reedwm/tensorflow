# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the loss scaling optimizer class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.util.tf_export import keras_export


class _UnwrapPreventer(object):
  """Wrapper that DistributionStrategy will not unwrap.

  Typically, DistributionStrategy will unwrap values when going from a cross-
  replica context to a replica context via `call_for_each_replica`. This class
  is a wrapper that DistributionStrategy will not unwrap, so it can be used to
  prevent it from unwrapping a value.

  TODO(reedwm): Find/implement a better way of preventing values from being
  unwrapped by DistributionStrategy
  """

  def __init__(self, value):
    self.value = value


@keras_export('keras.mixed_precision.experimental.LossScaleOptimizer')
class LossScaleOptimizer(optimizer_v2.OptimizerV2):
  """An optimizer that applies loss scaling.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. This is
  required to avoid numerical underflow when training with mixed precision on a
  GPU. See `tf.train.experimental.LossScale` for more information on loss
  scaling.

  This class takes another optimizer in it's constructor, and has the behavior
  of the optimizer, adding loss scaling. The original optimizer is not wrapped
  by this class, and this class holds no reference to the original optimizer.

  Loss scaling is applied whenever gradients are computed, either through
  `minimize()` or `get_gradients()`. The loss scale is updated via
  `tf.train.experimental.LossScale.update()` whenever gradients are applied,
  either through `minimize()` or `apply_gradients()`. For example:

  ```python
  opt = tf.keras.optimizers.SGD(0.1)
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
  # 'minimize' applies loss scaling to the loss and updates the loss sale.
  opt.minimize(loss_fn)
  ```

  If a `tf.GradientTape` is used to compute gradients instead of
  `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, the loss
  and gradients must be scaled manually. This can be done by calling
  `LossScaleOptimizer.get_scaled_loss` before passing the loss to
  `tf.GradientTape`, and `LossScaleOptimizer.get_unscaled_gradients` after
  computing the gradients with `tf.GradientTape`. For example:

  ```python
  opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(...)
  vars = ...
  with tf.GradientTape() as tape:
    loss = ...
    scaled_loss = opt.get_scaled_loss(loss)
  scaled_grads = tape.gradient(scaled_loss, vars)
  grads = opt.get_unscaled_gradients(scaled_grads)
  opt.apply_gradients(zip(grads, vars))  # Loss scale will be updated here
  ```

  The loss scale optimizer will not share slot variables with the old optimizer,
  but will instead recreate the slot variables and reinitialize them from
  scratch. The only state from the old optimizer that is retained is the
  information in `optimizer.get_config()`.

  #### Implementation note
  To emulate the original optimizer, the [three
  argument type function](https://docs.python.org/3/library/functions.html#type)
  is used to dynamically create a new type. This type subclasses both from
  LossScaleOptimizer and the original optimizer's type.

  LossScaleOptimizer.__new__ is overridden to return an instance of the
  dynamically created type, instead of directly returning a LossScaleOptimizer.
  This means that `type(LossScaleOptimizer(...)) != LossScaleOptimizer`,
  although `isinstance(LossScaleOptimizer(...), LossScaleOptimizer)` will still
  be true.

  The fact LossScaleOptimizer subclasses from OptimizerV2 has no effect, as
  the dynamic type will always subclass from the original optimizer's type,
  itself which subclasses OptimizerV2. LossScaleOptimizer only subclasses
  OptimizerV2 for documentation purposes.
  """

  @staticmethod
  def _define_loss_scale_optimizer_type(base_class):
    """Returns a subclass of `base_class` that implements loss scaling.

    This function dynamically creates a type that subclasses from both
    LossScaleOptimizer and `base_class`. `base_class` must be a subclass of
    OptimizerV2. The dynamic type will have the same functionality of
    `base_class` except loss scaling will be done.

    Args:
      base_class: A type that is a subclass of OptimizerV2

    Returns:
      A new type that is a subclass of `base_class`.
    """
    assert issubclass(base_class, optimizer_v2.OptimizerV2), (
        'base_class should be subclass of OptimizerV2 but got: %s' % base_class)

    # The dynamic type should not call `LossScaleOptimizer.__new__` when
    # constructed, despite inheriting from LossScaleOptimizer. This is because
    # `LossScaleOptimizer.__new__` itself calls
    # `_define_loss_scale_optimizer_type`, which would cause infinite recursion.
    # So we override the dynamic type's __new__ to call the base class's
    # __new__.
    def new(inner_cls, loss_scale, **kwargs):
      del loss_scale, kwargs
      # TODO(reedwm): If base_class overrides __new__ with a version that takes
      # more than 1 argument, this call will fail.
      obj = base_class.__new__(inner_cls)
      object.__setattr__(obj, '_is_initialized', False)
      return obj

    return type(
        'LossScaleOptimizer',
        (LossScaleOptimizer, base_class,),
        {'__new__': new}
    )

  def __new__(cls, opt, loss_scale):
    """Creates a new LossScaleOptimizer.

    This does not return a direct instance of LossScaleOptimizer, but instead
    a dynamically-created subclass of LossScaleOptimizer. This dynamic type
    subclasses from both LossScaleOptimizer and `optimizer`'s type, and it
    provides the functionality of `optimizer`, adding loss scaling on type of
    it.

    Args:
      opt: A `tf.keras.optimizers.Optimizer`. The LossScaleOptimizer will have
        the same behavior as this optimizer, but loss scaling will be added.
      loss_scale: The loss scale to scale the loss and gradients. This can
        either be an int/float to use a fixed loss scale, the string "dynamic"
        to use dynamic loss scaling, or an instance of a
        `tf.train.experimental.LossScale`. "dynamic" is recommended.

    Returns:
      An optimizer that subclasses both from `optimizer`'s type and
      LossScaleOptimizer.
    """
    if not isinstance(opt, optimizer_v2.OptimizerV2):
      raise ValueError('"opt" must be an instance of a '
                       'tf.keras.optimizers.Optimizer, but got: %s' %
                       (opt,))
    if isinstance(opt, LossScaleOptimizer):
      raise ValueError('Cannot create a LossScaleOptimizer from an existing '
                       'LossScaleOptimizer: %s' % (opt,))

    # __new__ consists of four steps:
    # Step 1: Define the type that will be returned from __new__. This is a
    # subclass of both LossScaleOptimizer and the original optimizer's class
    # (optimizer.__class__).
    lso_type = cls._define_loss_scale_optimizer_type(opt.__class__)
    # Step 2: Get the config of the original optimizer, and add loss scaling.
    # The 'opt' config is ignored by __init__, but we need to pass something
    # (alternatively we could pass None or anything else instead of optimizer)
    config = opt.get_config()
    config['loss_scale'] = loss_scale
    config['opt'] = opt
    # Step 3: Construct the new optimizer by calling the `from_config` method of
    # the original optimizer. This reconstructs all the state of the original
    # optimizer, plus adds the 'loss_scale' argument. We need the super() call,
    # because `from_config` of `lso_type` ends up calling this method (__new__)
    # and we need to avoid infinite recursion.
    obj = super(LossScaleOptimizer, lso_type).from_config(config)
    # Step 4: Set the _base_class attribute. This is used to get the base class
    # name in LossScaleOptimizer.get_config, so that the base class can be used
    # in LossScaleOptimizer.from_config to dynamically create a class with the
    # correct base class.
    obj._base_class = opt.__class__  # pylint: disable=protected-access
    return obj

  def __init__(self, opt, loss_scale, **kwargs):
    """Initializes this LossScaleOptimizer.

    Args:
      opt: A `tf.keras.optimizers.Optimizer`. The LossScaleOptimizer will have
        the same behavior as this optimizer, but loss scaling will be added.
      loss_scale: The loss scale to scale the loss and gradients. This can
        either be an int/float to use a fixed loss scale, the string "dynamic"
        to use dynamic loss scaling, or an instance of a
        `tf.train.experimental.LossScale`. "dynamic" is recommended.
      **kwargs: Must be empty. For internal use only.
    """
    # We have to check if the LossScaleOptimizer is already initialized.
    # This class is extremely unusual as __init__ will be called twice:
    #  1. __init__ is called inside __new__, as `from_config` is called within
    #     __new__, which constructs a LossScaleOptimizer. In this case, __init__
    #     will be called with the `opt` argument, the 'loss_scale' argument, and
    #     additional arguments to the base optimizer. `optimizer` is ignored.
    #  2. __init__ is called immediately after __new__, as Python automatically
    #     calls __new__ then __init__ when calling a class object (e.g., when
    #     running `LossScaleOptimizer(opt, 'dynamic')`). In this case, __init__
    #     will be called with only the 'opt' and 'loss_scale' arguments, as
    #     documented by the docstring. We ignore such arguments and skip
    #     initializing a second time, as the two arguments are used in __new__
    #     instead.
    # We only want to initialize the LossScaleOptimizer once, so we have this
    # special check.
    del opt
    is_initialized = object.__getattribute__(self, '_is_initialized')
    if is_initialized:
      return
    object.__setattr__(self, '_is_initialized', True)

    self._init(loss_scale, **kwargs)

  def _init(self, loss_scale, *args, **kwargs):
    """Initializes this LSO, without checking if already initialized."""
    super(LossScaleOptimizer, self).__init__(*args, **kwargs)
    if hasattr(self, '_loss_scale'):
      raise ValueError('_loss_scale is already defined')
    self._loss_scale = loss_scale_module.get(loss_scale)
    for weight in loss_scale_module.get_loss_scale_weights(self._loss_scale):
      # We cannot call `track_variable` in the LossScale class itself, because a
      # file outside of Keras cannot depend on a Keras file. Calling it here
      # instead is OK, because a variable only needs to be tracked if used with
      # a Keras class, and the only way to use LossScale with a Keras class is
      # through the LossScaleOptimizer.
      backend.track_variable(weight)
    self._track_trackable(self._loss_scale, 'loss_scale')

  @property
  def loss_scale(self):
    """The `LossScale` instance associated with this optimizer."""
    return self._loss_scale

  def get_scaled_loss(self, loss):
    """Scales the loss by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to scale the loss before
    passing the loss to `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_unscaled_gradients` should also be called.
    See the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for
    an example.

    Args:
      loss: The loss, which will be multiplied by the loss scale. Can either be
        a tensor or a callable returning a tensor.

    Returns:
      `loss` multiplied by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    if callable(loss):
      def new_loss():
        loss_val = loss()
        return loss_val * math_ops.cast(loss_scale, loss_val.dtype)
      return new_loss
    else:
      return loss * math_ops.cast(loss_scale, loss.dtype)

  def get_unscaled_gradients(self, grads):
    """Unscales the gradients by the loss scale.

    This method is only needed if you compute gradients manually, e.g. with
    `tf.GradientTape`. In that case, call this method to unscale the gradients
    after computing them with `tf.GradientTape`. If you use
    `LossScaleOptimizer.minimize` or `LossScaleOptimizer.get_gradients`, loss
    scaling is automatically applied and this method is unneeded.

    If this method is called, `get_scaled_loss` should also be called. See
    the `tf.keras.mixed_precision.experimental.LossScaleOptimizer` doc for an
    example.

    Args:
      grads: A list of tensors, each which will be divided by the loss scale.
        Can have None values, which are ignored.

    Returns:
      A new list the same size as `grads`, where every non-None value in `grads`
      is divided by `LossScaleOptimizer.loss_scale()`.
    """
    loss_scale = self._loss_scale()
    loss_scale_reciprocal = 1. / loss_scale
    return [g * math_ops.cast(loss_scale_reciprocal, g.dtype) if g is not None
            else None for g in grads]

  def _compute_gradients(self, loss, var_list, grad_loss=None):
    loss = self.get_scaled_loss(loss)
    grads_and_vars = super(LossScaleOptimizer, self)._compute_gradients(
        loss, var_list, grad_loss)
    grads = [g for g, _ in grads_and_vars]
    variables = [v for _, v in grads_and_vars]
    unscaled_grads = self.get_unscaled_gradients(grads)
    return list(zip(unscaled_grads, variables))

  def get_gradients(self, loss, params):
    loss = self.get_scaled_loss(loss)
    grads = super(LossScaleOptimizer, self).get_gradients(loss, params)
    return self.get_unscaled_gradients(grads)

  def apply_gradients(self, grads_and_vars, name=None):
    if distribution_strategy_context.in_cross_replica_context():
      raise ValueError('apply_gradients() must be called in a replica context.')
    grads_and_vars = tuple(grads_and_vars)
    return distribution_strategy_context.get_replica_context().merge_call(
        self._apply_gradients_cross_replica, args=(grads_and_vars, name))

  def _apply_gradients_cross_replica(self, distribution, grads_and_vars, name):
    """Apply gradients in a cross replica context."""
    grads = [g for g, _ in grads_and_vars]
    loss_scale_update_op, should_apply_grads = self._loss_scale.update(grads)

    def apply_fn():
      # We do not want DistributionStrategy to unwrap any MirroredVariables in
      # grads_and_vars, because even in a replica context, the wrapped optimizer
      # expects mirrored variables. So we wrap the variables with an
      # _UnwrapPreventer, preventing DistributionStrategy from unwrapping the
      # MirroredVariables.
      wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])
      return distribution.extended.call_for_each_replica(
          self._apply_gradients, args=(grads, wrapped_vars, name))

    # Note: We must call this cond() in a cross-replica context.
    # DistributionStrategy does not support having a cond in a replica context
    # with a branch that calls `merge_call`, and self._optimizer.apply_gradients
    # calls `merge_call`.
    maybe_apply_op = smart_cond.smart_cond(should_apply_grads,
                                           apply_fn,
                                           control_flow_ops.no_op)
    return control_flow_ops.group(maybe_apply_op, loss_scale_update_op)

  def _apply_gradients(self, grads, wrapped_vars, name):
    return super(LossScaleOptimizer, self).apply_gradients(
        list(zip(grads, wrapped_vars.value)), name)

  # TODO(reedwm): Maybe merge this class's functionality into OptimizerV2.

  # TODO(reedwm): Maybe throw an error if mixed precision is used without this
  # optimizer being used.

  def get_config(self):
    return {
        'base_optimizer_class_name': self._base_class.__name__,
        'base_optimizer_config': super(LossScaleOptimizer, self).get_config(),
        'loss_scale': self.loss_scale
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    """Creates a LossScaleOptimizer from it's config."""
    config = dict(config)  # Make a copy, since we mutate config
    base_config = {
        'class_name': config.pop('base_optimizer_class_name'),
        'config': config.pop('base_optimizer_config')
    }
    base_opt = optimizers.deserialize(base_config,
                                      custom_objects=custom_objects)
    # We call `LossScaleOptimizer` instead of `cls`, because only
    # `LossScaleOptimizer.__new__` handles taking in an optimizer and
    # constructing a new class. `cls` might be a LossScaleOptimizer, but could
    # also be a dynamic subclass of LossScaleOptimizer defined in
    # _define_loss_scale_optimizer_type(). The dynamic subclass's __new__ method
    # does not take in an optimizer as input, so it cannot be used.
    return LossScaleOptimizer(base_opt, **config)
