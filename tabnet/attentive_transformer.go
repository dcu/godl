package tabnet

import (
	"math"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	InputDimension                   int
	OutputDimension                  int
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	Inferring                        bool
	Activation                       godl.ActivationFn
	WithBias                         bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *AttentiveTransformerOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = godl.Sparsemax
	}

	if o.WeightsInit == nil {
		gain := math.Sqrt(float64(o.InputDimension+o.OutputDimension) / math.Sqrt(float64(4*o.InputDimension)))
		o.WeightsInit = gorgonia.GlorotN(gain)
	}
}

// AttentiveTransformer implements an attetion transformer layer
func AttentiveTransformer(nn *godl.Model, opts AttentiveTransformerOpts) godl.Layer {
	lt := godl.AddLayer("AttentiveTransformer")

	opts.setDefaults()

	fcLayer := godl.FC(nn, godl.FCOpts{
		InputDimension:  opts.InputDimension,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     opts.WeightsInit,
		WithBias:        opts.WithBias,
	})

	gbnLayer := godl.GBN(nn, godl.GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	return func(nodes ...*gorgonia.Node) (godl.Result, error) {
		if err := nn.CheckArity(lt, nodes, 2); err != nil {
			return godl.Result{}, err
		}

		x := nodes[0]
		prior := nodes[1]

		fc, err := fcLayer(x)
		if err != nil {
			return godl.Result{}, godl.ErrorF(lt, "fc%v failed failed: %w", x.Shape(), err)
		}

		bn, err := gbnLayer(fc.Output)
		if err != nil {
			return godl.Result{}, godl.ErrorF(lt, "gbn%v failed: %w", fc.Shape(), err)
		}

		mul, err := gorgonia.HadamardProd(bn.Output, prior)
		if err != nil {
			return godl.Result{}, godl.ErrorF(lt, "hadamardProd(%v, %v) failed: %w", bn.Shape(), prior.Shape(), err)
		}

		sm, err := opts.Activation(mul)
		if err != nil {
			return godl.Result{}, godl.ErrorF(lt, "fn(%v) failed: %w", mul.Shape(), err)
		}

		return godl.Result{Output: sm}, nil
	}
}
