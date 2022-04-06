package godl

import (
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
)

var gluCount uint64 = 0

// GLUOpts are the supported options for GLU
type GLUOpts struct {
	InputDimension   int
	OutputDimension  int
	VirtualBatchSize int
	Activation       activation.Function
	Linear           *LinearModule
	WeightsInit      gorgonia.InitWFn
	WithBias         bool
	Momentum         float64
}

func (opts *GLUOpts) setDefaults() {
	if opts.Momentum == 0 {
		opts.Momentum = 0.02
	}

	if opts.Activation == nil {
		opts.Activation = gorgonia.Sigmoid
	}

	if opts.InputDimension == 0 {
		panic("input dimension must be set")
	}

	if opts.OutputDimension == 0 {
		panic("output dimension must be set")
	}

	if opts.VirtualBatchSize == 0 {
		panic("virtual batch size must be set")
	}
}

type GLUModule struct {
	model  *Model
	layer  LayerType
	opts   GLUOpts
	gbn    *GhostBatchNormModule
	linear *LinearModule
}

func (m *GLUModule) Forward(inputs ...*Node) Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]

	fcResult := m.opts.Linear.Forward(x)
	gnbResult := m.gbn.Forward(fcResult...)[0]

	// GLU
	firstHalf := gorgonia.Must(gorgonia.Slice(gnbResult, nil, gorgonia.S(0, m.opts.OutputDimension)))
	secondHalf := gorgonia.Must(gorgonia.Slice(gnbResult, nil, gorgonia.S(m.opts.OutputDimension, gnbResult.Shape()[1])))

	act, err := m.opts.Activation(secondHalf)
	if err != nil {
		panic(ErrorF(m.layer, "%s: applying activation function failed: %w", err))
	}

	mul, err := gorgonia.HadamardProd(firstHalf, act)
	if err != nil {
		panic(ErrorF(m.layer, "%s: HadamardProd %d x %d: %w", firstHalf.Shape(), act.Shape(), err))
	}

	return Nodes{mul}
}

// GLU implements a Gated Linear Unit Block
func GLU(nn *Model, opts GLUOpts) *GLUModule {
	opts.setDefaults()

	lt := AddLayer("GLU")

	if opts.Linear == nil {
		opts.Linear = Linear(nn, LinearOpts{
			InputDimension:  opts.InputDimension,
			OutputDimension: opts.OutputDimension * 2,
			WeightsInit:     opts.WeightsInit,
			WithBias:        opts.WithBias,
		})
	}

	gbn := GhostBatchNorm(nn, GhostBatchNormOpts{
		VirtualBatchSize: opts.VirtualBatchSize,
		OutputDimension:  opts.OutputDimension * 2,
		Momentum:         opts.Momentum,
	})

	return &GLUModule{
		model:  nn,
		layer:  lt,
		opts:   opts,
		gbn:    gbn,
		linear: opts.Linear,
	}
}
