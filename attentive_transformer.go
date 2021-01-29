package tabnet

import (
	"math"

	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	InputDimension                   int
	OutputDimension                  int
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	Inferring                        bool
	Activation                       ActivationFn
	WithBias                         bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *AttentiveTransformerOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = Sparsemax
	}

	if o.WeightsInit == nil {
		gain := math.Sqrt(float64(o.InputDimension+o.OutputDimension) / math.Sqrt(float64(4*o.InputDimension)))
		o.WeightsInit = gorgonia.GlorotN(gain)
	}
}

// AttentiveTransformer implements an attetion transformer layer
func AttentiveTransformer(nn *Model, opts AttentiveTransformerOpts) Layer {
	lt := AddLayer("AttentiveTransformer")

	opts.setDefaults()

	fcLayer := FC(nn, FCOpts{
		InputDimension:  opts.InputDimension,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     opts.WeightsInit,
		WithBias:        opts.WithBias,
	})

	gbnLayer := GBN(nn, GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension,
		Inferring:        opts.Inferring,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	return func(nodes ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		if err := nn.CheckArity(lt, nodes, 2); err != nil {
			return nil, nil, err
		}

		x := nodes[0]
		prior := nodes[1]

		fc, _, err := fcLayer(x)
		if err != nil {
			return nil, nil, errorF(lt, "fc%v failed failed: %w", x.Shape(), err)
		}

		bn, _, err := gbnLayer(fc)
		if err != nil {
			return nil, nil, errorF(lt, "gbn%v failed: %w", fc.Shape(), err)
		}

		mul, err := gorgonia.HadamardProd(bn, prior)
		if err != nil {
			return nil, nil, errorF(lt, "hadamardProd(%v, %v) failed: %w", bn.Shape(), prior.Shape(), err)
		}

		sm, err := opts.Activation(mul)
		if err != nil {
			return nil, nil, errorF(lt, "fn(%v) failed: %w", mul.Shape(), err)
		}

		return sm, nil, nil
	}
}
