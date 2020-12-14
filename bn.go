package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// BNOpts are the options to configure a batch normalization
type BNOpts struct {
	Momentum            float64
	Epsilon             float64
	Inferring           bool
	ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *BNOpts) setDefaults() {
	if o.Momentum == 0.0 {
		o.Momentum = 0.01
	}

	if o.Epsilon == 0.0 {
		o.Epsilon = 1e-5
	}

	if o.ScaleInit == nil {
		o.ScaleInit = gorgonia.Ones()
	}

	if o.BiasInit == nil {
		o.BiasInit = gorgonia.Zeroes()
	}
}

// BN runs a batch normalization on the input x
func (nn *Model) BN(opts BNOpts) Layer {
	opts.setDefaults()

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, error) {
		x := nodes[0]
		xShape := x.Shape()
		if xShape.Dims() == 1 {
			x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{1, 1, 1, xShape[0]}))
		}

		scale := nn.addLearnable("scale", x.Shape(), opts.ScaleInit)
		bias := nn.addBias(x.Shape(), opts.BiasInit)

		ret, _, _, bnop, err := gorgonia.BatchNorm(x, scale, bias, opts.Momentum, opts.Epsilon)
		if err != nil {
			return nil, err
		}

		if opts.Inferring {
			bnop.SetTesting()
		} else {
			bnop.SetTraining()
		}

		return gorgonia.Reshape(ret, xShape)
	}
}
