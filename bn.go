package tabnet

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// BNOpts are the options to configure a batch normalization
type BNOpts struct {
	Momentum            float64
	Epsilon             float64
	Inferring           bool
	ScaleInit, BiasInit gorgonia.InitWFn

	InputDim  int
	OutputDim int
}

func (o *BNOpts) setDefaults() {
	if o.InputDim == 0 {
		panic("input size for BN can't be 0")
	}

	if o.OutputDim == 0 {
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

// BN runs a batch normalization on the input x
func (nn *Model) BN(opts BNOpts) Layer {
	opts.setDefaults()

	lt := incLayer("BN")

	bias := nn.addBias(lt, tensor.Shape{opts.InputDim, 1}, opts.BiasInit)
	scale := nn.addLearnable(lt, "scale", tensor.Shape{opts.InputDim, 1}, opts.ScaleInit)

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		if err := nn.checkArity(lt, nodes, 1); err != nil {
			return nil, nil, err
		}

		x := nodes[0]

		ret, _, _, bnop, err := gorgonia.BatchNorm1d(x, scale, bias, opts.Momentum, opts.Epsilon)
		if err != nil {
			return nil, nil, fmt.Errorf("BatchNorm1d: %w", err)
		}

		if opts.Inferring {
			bnop.SetTesting()
		} else {
			bnop.SetTraining()
		}

		return ret, nil, nil
	}
}
