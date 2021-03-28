package godl

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// BatchNormOpts are the options to configure a batch normalization
type BatchNormOpts struct {
	Momentum            float32
	Epsilon             float32
	ScaleInit, BiasInit gorgonia.InitWFn

	InputSize int
}

func (o *BatchNormOpts) setDefaults() {
	if o.InputSize == 0 {
		panic("output size for BN can't be 0")
	}

	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.ScaleInit == nil {
		o.ScaleInit = gorgonia.Ones()
		// gain := math.Sqrt(float64(o.InputDim+o.OutputDim) / math.Sqrt(float64(4*o.InputDim)))
		// o.ScaleInit = gorgonia.GlorotN(gain)
	}

	if o.BiasInit == nil {
		o.BiasInit = gorgonia.Zeroes()
	}
}

// BatchNorm1d defines the batch norm operation for tensors with shape (B, N)
func BatchNorm1d(nn *Model, opts BatchNormOpts) Layer {
	opts.setDefaults()
	lt := AddLayer("BatchNorm1d")

	scale := nn.AddLearnable(lt, "scale", tensor.Shape{1, opts.InputSize}, NewWeightsOpts{
		InitFN: opts.ScaleInit,
	})
	bias := nn.AddBias(lt, tensor.Shape{1, opts.InputSize}, NewWeightsOpts{
		InitFN: opts.BiasInit,
	})

	return batchNorm(nn, lt, scale, bias, opts)
}

// BatchNorm2d defines the batch norm operation for tensors with shape (B, C, W, H)
func BatchNorm2d(nn *Model, opts BatchNormOpts) Layer {
	opts.setDefaults()
	lt := AddLayer("BatchNorm2d")

	scale := nn.AddLearnable(lt, "scale", tensor.Shape{1, opts.InputSize, 1, 1}, NewWeightsOpts{
		InitFN: opts.ScaleInit,
	})
	bias := nn.AddBias(lt, tensor.Shape{1, opts.InputSize, 1, 1}, NewWeightsOpts{
		InitFN: opts.BiasInit,
	})

	return batchNorm(nn, lt, scale, bias, opts)
}

// batchNorm runs a batch normalization on the input x
func batchNorm(nn *Model, lt LayerType, scale, bias *gorgonia.Node, opts BatchNormOpts) Layer {
	opts.setDefaults()

	return func(nodes ...*gorgonia.Node) (Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return Result{}, err
		}

		x := nodes[0]

		ret, _, _, bnop, err := gorgonia.BatchNorm(x, scale, bias, float64(opts.Momentum), float64(opts.Epsilon))
		if err != nil {
			return Result{}, fmt.Errorf("%v: %w", lt, err)
		}

		if !nn.Training {
			bnop.SetTesting()
		} else {
			bnop.SetTraining()
		}

		return Result{Output: ret}, nil
	}
}
