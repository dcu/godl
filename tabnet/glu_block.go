package tabnet

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
)

type GLUBlockOpts struct {
	InputDimension   int
	OutputDimension  int
	Shared           []*godl.LinearModule
	VirtualBatchSize int

	Size int

	WithBias    bool
	Momentum    float64
	WeightsInit gorgonia.InitWFn
}

type GLUBlockModule struct {
	model *godl.Model
	layer godl.LayerType
	opts  GLUBlockOpts

	gluLayers []*godl.GLUModule
	scale     *godl.Node
}

func (m *GLUBlockModule) Forward(inputs ...*godl.Node) godl.Nodes {
	if err := m.model.CheckArity(m.layer, inputs, 1); err != nil {
		panic(err)
	}

	x := inputs[0]
	startAt := 0

	if len(m.opts.Shared) > 0 {
		result := m.gluLayers[0].Forward(x)

		x = result[0]
		startAt = 1
	}

	for _, glu := range m.gluLayers[startAt:] {
		result := glu.Forward(x)[0]

		x = gorgonia.Must(gorgonia.Add(x, result))
		x = gorgonia.Must(gorgonia.Mul(x, m.scale))
	}

	return godl.Nodes{x}
}

func GLUBlock(nn *godl.Model, opts GLUBlockOpts) *GLUBlockModule {
	lt := godl.AddLayer("GLUBlock")

	gluLayers := make([]*godl.GLUModule, 0, opts.Size)
	gluInput := opts.InputDimension
	if len(opts.Shared) == 0 { // for independent layers
		gluInput = opts.OutputDimension
	}

	gluOutput := opts.OutputDimension
	weightsInit := opts.WeightsInit

	if weightsInit == nil {
		gain := math.Sqrt(float64(gluInput+gluOutput) / math.Sqrt(float64(gluInput)))
		weightsInit = gorgonia.GlorotN(gain)
	}

	for i := 0; i < opts.Size; i++ {
		var fcLayer *godl.LinearModule
		if len(opts.Shared) > 0 {
			fcLayer = opts.Shared[i]
		}

		gluLayers = append(gluLayers, godl.GLU(nn, godl.GLUOpts{
			InputDimension:   gluInput,
			OutputDimension:  gluOutput,
			VirtualBatchSize: opts.VirtualBatchSize,
			Linear:           fcLayer,
			WeightsInit:      weightsInit,
			WithBias:         opts.WithBias,
			Momentum:         opts.Momentum,
		}))

		gluInput = gluOutput
	}

	scale := gorgonia.NewConstant(math32.Sqrt(0.5), gorgonia.WithName("ft.scale"))

	return &GLUBlockModule{
		model:     nn,
		layer:     lt,
		opts:      opts,
		gluLayers: gluLayers,
		scale:     scale,
	}
}
