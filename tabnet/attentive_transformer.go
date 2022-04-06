package tabnet

import (
	"math"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
)

type AttentiveTransformerOpts struct {
	InputDimension                   int
	OutputDimension                  int
	Momentum                         float64
	Epsilon                          float64
	VirtualBatchSize                 int
	Activation                       activation.Function
	WithBias                         bool
	WeightsInit, ScaleInit, BiasInit gorgonia.InitWFn
}

func (o *AttentiveTransformerOpts) setDefaults() {
	if o.Activation == nil {
		o.Activation = activation.SparseMax
	}

	if o.WeightsInit == nil {
		gain := math.Sqrt(float64(o.InputDimension+o.OutputDimension) / math.Sqrt(float64(4*o.InputDimension)))
		o.WeightsInit = gorgonia.GlorotN(gain)
	}
}

type AttentiveTransformerModule struct {
	model  *godl.Model
	layer  godl.LayerType
	opts   AttentiveTransformerOpts
	linear *godl.LinearModule
	gbn    *godl.GhostBatchNormModule
}

func (m *AttentiveTransformerModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 2); err != nil {
		panic(err)
	}

	x := inputs[0]
	prior := inputs[1]

	fc := m.linear.Forward(x)
	bn := m.gbn.Forward(fc...)

	mul := gorgonia.Must(gorgonia.HadamardProd(bn[0], prior))

	sm := gorgonia.Must(m.opts.Activation(mul))

	return godl.Nodes{sm}
}

// AttentiveTransformer implements an attetion transformer layer
func AttentiveTransformer(nn *godl.Model, opts AttentiveTransformerOpts) *AttentiveTransformerModule {
	lt := godl.AddLayer("AttentiveTransformer")

	opts.setDefaults()

	weightsInit := opts.WeightsInit
	if weightsInit == nil {
		gain := math.Sqrt(float64(opts.InputDimension+opts.OutputDimension) / math.Sqrt(float64(4*opts.InputDimension)))
		weightsInit = gorgonia.GlorotN(gain)
	}

	fcLayer := godl.Linear(nn, godl.LinearOpts{
		InputDimension:  opts.InputDimension,
		OutputDimension: opts.OutputDimension,
		WeightsInit:     weightsInit,
		WithBias:        opts.WithBias,
	})

	gbnLayer := godl.GhostBatchNorm(nn, godl.GhostBatchNormOpts{
		Momentum:         opts.Momentum,
		Epsilon:          opts.Epsilon,
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension,
		ScaleInit:        opts.ScaleInit,
		BiasInit:         opts.BiasInit,
	})

	return &AttentiveTransformerModule{
		model:  nn,
		layer:  lt,
		opts:   opts,
		linear: fcLayer,
		gbn:    gbnLayer,
	}
}
