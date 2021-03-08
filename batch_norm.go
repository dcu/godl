package deepzen

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

	InputDimension int
}

func (o *BatchNormOpts) setDefaults() {
	if o.InputDimension == 0 {
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

// BatchNorm runs a batch normalization on the input x
func BatchNorm(nn *Model, opts BatchNormOpts) Layer {
	opts.setDefaults()

	lt := AddLayer("BN")

	scale := nn.AddLearnable(lt, "scale", tensor.Shape{1, opts.InputDimension}, NewWeightsOpts{
		InitFN: opts.ScaleInit,
	})
	bias := nn.AddBias(lt, tensor.Shape{1, opts.InputDimension}, NewWeightsOpts{
		InitFN: opts.BiasInit,
	})

	return func(nodes ...*gorgonia.Node) (Result, error) {
		if err := nn.CheckArity(lt, nodes, 1); err != nil {
			return Result{}, err
		}

		x := nodes[0]

		bnFunc := gorgonia.BatchNorm1d
		if x.Dims() == 4 {
			bnFunc = gorgonia.BatchNorm
		}

		ret, _, _, bnop, err := bnFunc(x, scale, bias, float64(opts.Momentum), float64(opts.Epsilon))
		if err != nil {
			return Result{}, fmt.Errorf("BatchNorm1d: %w", err)
		}

		if !nn.Training {
			bnop.SetTesting()
		} else {
			bnop.SetTraining()
		}

		return Result{Output: ret}, nil
	}
}
