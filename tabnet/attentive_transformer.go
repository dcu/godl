package tabnet

import (
	"math"

	"github.com/dcu/deepzen"
	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	InputDimension                   int
	OutputDimension                  int
	Momentum                         float32
	Epsilon                          float32
	VirtualBatchSize                 int
	Inferring                        bool
	Activation                       deepzen.ActivationFn
	WithBias                         bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *AttentiveTransformerOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = deepzen.Sparsemax
	}

	if o.WeightsInit == nil {
		gain := math.Sqrt(float64(o.InputDimension+o.OutputDimension) / math.Sqrt(float64(4*o.InputDimension)))
		o.WeightsInit = gorgonia.GlorotN(gain)
	}
}

// AttentiveTransformer implements an attetion transformer layer
func AttentiveTransformer(nn *deepzen.Model, opts AttentiveTransformerOpts) deepzen.Layer {
	lt := deepzen.AddLayer("AttentiveTransformer")

	opts.setDefaults()

	fcLayer := deepzen.FC(nn, deepzen.FCOpts{
		InputDimension:  opts.InputDimension,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     opts.WeightsInit,
		WithBias:        opts.WithBias,
	})

	gbnLayer := deepzen.GBN(nn, deepzen.GBNOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	return func(nodes ...*gorgonia.Node) (deepzen.Result, error) {
		if err := nn.CheckArity(lt, nodes, 2); err != nil {
			return deepzen.Result{}, err
		}

		x := nodes[0]
		prior := nodes[1]

		fc, err := fcLayer(x)
		if err != nil {
			return deepzen.Result{}, deepzen.ErrorF(lt, "fc%v failed failed: %w", x.Shape(), err)
		}

		bn, err := gbnLayer(fc.Output)
		if err != nil {
			return deepzen.Result{}, deepzen.ErrorF(lt, "gbn%v failed: %w", fc.Shape(), err)
		}

		mul, err := gorgonia.HadamardProd(bn.Output, prior)
		if err != nil {
			return deepzen.Result{}, deepzen.ErrorF(lt, "hadamardProd(%v, %v) failed: %w", bn.Shape(), prior.Shape(), err)
		}

		sm, err := opts.Activation(mul)
		if err != nil {
			return deepzen.Result{}, deepzen.ErrorF(lt, "fn(%v) failed: %w", mul.Shape(), err)
		}

		return deepzen.Result{Output: sm}, nil
	}
}
