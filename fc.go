package godl

import (
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// LinearOpts contains optional parameter for a layer
type LinearOpts struct {
	Activation      activation.Function
	Dropout         float64
	OutputDimension int
	InputDimension  int

	WeightsInit           gorgonia.InitWFn
	BiasInit              gorgonia.InitWFn
	WithBias              bool
	WeightsName, BiasName string
	FixedWeights          bool
}

type LinearModule struct {
	model *Model
	opts  LinearOpts
	layer LayerType

	weight, bias *Node
}

func (m *LinearModule) Name() string {
	return "Linear"
}

func (m *LinearModule) Forward(inputs ...*Node) (out Nodes) {
	x := inputs[0]
	xShape := x.Shape()

	if x.Dims() > 2 {
		b, v := xShape[0], tensor.Shape(xShape[1:]).TotalSize()
		x = gorgonia.Must(gorgonia.Reshape(x, tensor.Shape{b, v}))
	}

	wT := gorgonia.Must(gorgonia.Transpose(m.weight, 1, 0))

	result, err := gorgonia.Mul(x, wT)
	if err != nil {
		panic(ErrorF(m.layer, "error applying mul %v x %v: %w ", x.Shape(), wT.Shape(), err))
	}

	if m.opts.WithBias {
		result, err = gorgonia.BroadcastAdd(result, m.bias, nil, []byte{0})
		if err != nil {
			panic(ErrorF(m.layer, "error adding bias %w", err))
		}
	}

	if m.opts.Activation != nil {
		result, err = m.opts.Activation(result)
		if err != nil {
			panic(ErrorF(m.layer, "error applying activation %w", err))
		}
	}

	if m.opts.Dropout > 0.0 {
		result, err = gorgonia.Dropout(result, m.opts.Dropout)
		if err != nil {
			panic(ErrorF(m.layer, "error applying dropout %w", err))
		}
	}

	return Nodes{result}
}

func Linear(nn *Model, opts LinearOpts) *LinearModule {
	lt := AddLayer("FC")

	MustBeGreatherThan(lt, "input dimension", opts.InputDimension, 0)
	MustBeGreatherThan(lt, "output dimension", opts.OutputDimension, 0)

	var (
		bias *gorgonia.Node
		w    = nn.AddWeights(lt, tensor.Shape{opts.OutputDimension, opts.InputDimension}, NewWeightsOpts{
			InitFN:     opts.WeightsInit,
			UniqueName: opts.WeightsName,
			Fixed:      opts.FixedWeights,
		})
	)

	if opts.WithBias {
		bias = nn.AddBias(lt, tensor.Shape{1, opts.OutputDimension}, NewWeightsOpts{
			InitFN:     opts.BiasInit,
			UniqueName: opts.BiasName,
			Fixed:      opts.FixedWeights,
		})
	}

	return &LinearModule{
		model:  nn,
		layer:  lt,
		opts:   opts,
		bias:   bias,
		weight: w,
	}
}

var (
	_ Module = &LinearModule{}
)
