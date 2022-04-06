package lstm

import (
	"fmt"

	"github.com/dcu/godl"
	"github.com/dcu/godl/activation"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MergeMode string

const (
	MergeModeConcat  MergeMode = "concat"
	MergeModeSum     MergeMode = "sum"
	MergeModeAverage MergeMode = "avg"
	MergeModeMul     MergeMode = "mul"
)

type LSTMOpts struct {
	InputDimension int
	HiddenSize     int
	// Layers         int
	Bidirectional bool
	MergeMode     MergeMode

	WithBias              bool
	WeightsInit, BiasInit gorgonia.InitWFn

	RecurrentActivation activation.Function
	Activation          activation.Function
}

func (o *LSTMOpts) setDefaults() {
	if o.RecurrentActivation == nil {
		o.RecurrentActivation = activation.Sigmoid
	}

	if o.Activation == nil {
		o.Activation = activation.Tanh
	}

	if o.MergeMode == "" {
		o.MergeMode = MergeModeConcat
	}
}

type LSTMModule struct {
	model   *godl.Model
	layer   godl.LayerType
	opts    LSTMOpts
	weights []lstmParams
}

func (m *LSTMModule) Forward(inputs ...*godl.Node) godl.Nodes {
	two := gorgonia.NewConstant(float32(2.0), gorgonia.WithName("two"))

	var (
		x, prevHidden, prevCell *gorgonia.Node
	)

	switch len(inputs) {
	case 1:
		x = inputs[0]
		batchSize := x.Shape()[1]

		dummyHidden := gorgonia.NewTensor(m.model.TrainGraph(), tensor.Float32, 3, gorgonia.WithShape(1, batchSize, m.opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyHidden")) // FIXME: unique name
		dummyCell := gorgonia.NewTensor(m.model.TrainGraph(), tensor.Float32, 3, gorgonia.WithShape(1, batchSize, m.opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyCell"))

		prevHidden = dummyHidden
		prevCell = dummyCell
	case 3:
		x = inputs[0]
		prevHidden = inputs[1]
		prevCell = inputs[2]
	default:
		panic(fmt.Errorf("%v: invalid input size", m.layer))
	}

	if !m.opts.Bidirectional {
		return lstm(m.model, x, m.weights[0].inputWeights, prevHidden, m.weights[0].hiddenWeights, prevCell, m.weights[0].bias, false, m.opts)
	}

	x1 := gorgonia.Must(gorgonia.BatchedMatMul(x, m.weights[0].inputWeights))
	forwardOutput := lstm(m.model, x1, m.weights[0].inputWeights, prevHidden, m.weights[0].hiddenWeights, prevCell, m.weights[0].bias, true, m.opts)

	x2 := gorgonia.Must(gorgonia.BatchedMatMul(x, m.weights[1].inputWeights))
	x2 = gorgonia.Must(gorgonia.ApplyOp(OrderingOp{}, x2))

	backwardOutput := lstm(m.model, x2, m.weights[1].inputWeights, prevHidden, m.weights[1].hiddenWeights, prevCell, m.weights[1].bias, true, m.opts)

	backwardOutputReversed := gorgonia.Must(gorgonia.ApplyOp(OrderingOp{}, backwardOutput[0]))

	var output *godl.Node

	switch m.opts.MergeMode {
	case MergeModeAverage:
		output = gorgonia.Must(gorgonia.Div(gorgonia.Must(gorgonia.Add(forwardOutput[0], backwardOutputReversed)), two))
	case MergeModeConcat:
		output = gorgonia.Must(gorgonia.Concat(forwardOutput[0].Dims()-1, forwardOutput[0], backwardOutputReversed))
	case MergeModeMul:
		output = gorgonia.Must(gorgonia.HadamardProd(forwardOutput[0], backwardOutputReversed))
	case MergeModeSum:
		output = gorgonia.Must(gorgonia.Add(forwardOutput[0], backwardOutputReversed))
	}

	hidden := gorgonia.Must(gorgonia.Concat(0, forwardOutput[1], backwardOutput[1]))
	cell := gorgonia.Must(gorgonia.Concat(0, forwardOutput[2], backwardOutput[2]))

	return godl.Nodes{output, hidden, cell}
}

func LSTM(m *godl.Model, opts LSTMOpts) *LSTMModule {
	opts.setDefaults()
	lt := godl.AddLayer("LSTM")

	paramsCount := 1
	if opts.Bidirectional {
		paramsCount = 2
	}

	weights := buildParamsList(paramsCount, m, lt, opts)

	return &LSTMModule{
		model:   m,
		layer:   lt,
		weights: weights,
		opts:    opts,
	}
}

type lstmParams struct {
	inputWeights, hiddenWeights, bias *gorgonia.Node
}

func buildParamsList(count int, m *godl.Model, lt godl.LayerType, opts LSTMOpts) []lstmParams {
	list := make([]lstmParams, count)
	for i := 0; i < count; i++ {
		list[i] = newParams(m, lt, opts)
	}

	return list
}

func newParams(m *godl.Model, lt godl.LayerType, opts LSTMOpts) lstmParams {
	inputWeightsSize := 1
	if opts.Bidirectional {
		inputWeightsSize = 2
	}

	inputWeights := m.AddWeights(lt, tensor.Shape{inputWeightsSize, opts.InputDimension, opts.HiddenSize * 4}, godl.NewWeightsOpts{
		InitFN: opts.WeightsInit,
	})
	hiddenWeights := m.AddWeights(lt, tensor.Shape{1, opts.HiddenSize, opts.HiddenSize * 4}, godl.NewWeightsOpts{
		InitFN: opts.WeightsInit,
	})

	var bias *gorgonia.Node

	if opts.WithBias {
		bias = m.AddBias(lt, tensor.Shape{1, 1, opts.HiddenSize * 4}, godl.NewWeightsOpts{
			InitFN: opts.BiasInit,
		})
	}

	return lstmParams{
		inputWeights:  inputWeights,
		hiddenWeights: hiddenWeights,
		bias:          bias,
	}
}

func lstm(m *godl.Model, x, inputWeights, prevHidden, hiddenWeights, prevCell, bias *gorgonia.Node, withPrecomputedInput bool, opts LSTMOpts) godl.Nodes {
	seqs := x.Shape()[0]
	outputs := make([]*gorgonia.Node, seqs)

	var err error

	for seq := 0; seq < seqs; seq++ {
		seqX := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(seq), nil, nil))
		seqX = gorgonia.Must(gorgonia.Reshape(seqX, tensor.Shape{1, seqX.Shape()[0], seqX.Shape()[1]}))

		prevHidden, prevCell, err = lstmGate(m, seqX, inputWeights, prevHidden, hiddenWeights, prevCell, bias, withPrecomputedInput, opts)
		if err != nil {
			panic(err)
		}

		outputs[seq] = prevHidden
	}

	outputGate := gorgonia.Must(gorgonia.Concat(0, outputs...))

	return godl.Nodes{outputGate, prevHidden, prevCell}
}

func lstmGate(m *godl.Model, seqX, inputWeights, prevHidden, hiddenWeights, prevCell, bias *gorgonia.Node, withPrecomputedInput bool, opts LSTMOpts) (*gorgonia.Node, *gorgonia.Node, error) {
	prevHidden = gorgonia.Must(gorgonia.BatchedMatMul(prevHidden, hiddenWeights))

	if !withPrecomputedInput {
		seqX = gorgonia.Must(gorgonia.BatchedMatMul(seqX, inputWeights))
	}

	gates := gorgonia.Must(gorgonia.Add(prevHidden, seqX))

	if bias != nil {
		gates = gorgonia.Must(gorgonia.BroadcastAdd(gates, bias, nil, []byte{0, 1}))
	}

	inputGate := gorgonia.Must(gorgonia.Slice(gates, nil, nil, gorgonia.S(0, opts.HiddenSize)))
	inputGate = gorgonia.Must(opts.RecurrentActivation(inputGate))

	forgetGate := gorgonia.Must(gorgonia.Slice(gates, nil, nil, gorgonia.S(opts.HiddenSize, opts.HiddenSize*2)))
	forgetGate = gorgonia.Must(opts.RecurrentActivation(forgetGate))

	cellGate := gorgonia.Must(gorgonia.Slice(gates, nil, nil, gorgonia.S(opts.HiddenSize*2, opts.HiddenSize*3)))
	cellGate = gorgonia.Must(opts.Activation(cellGate))

	outputGate := gorgonia.Must(gorgonia.Slice(gates, nil, nil, gorgonia.S(opts.HiddenSize*3, opts.HiddenSize*4)))
	outputGate = gorgonia.Must(opts.RecurrentActivation(outputGate))

	retain, err := gorgonia.BroadcastHadamardProd(forgetGate, prevCell, nil, []byte{0})
	if err != nil {
		return nil, nil, err
	}

	write, err := gorgonia.BroadcastHadamardProd(inputGate, cellGate, nil, []byte{0})
	if err != nil {
		return nil, nil, err
	}

	prevCell, err = gorgonia.Add(retain, write)
	if err != nil {
		return nil, nil, err
	}

	cellTan := gorgonia.Must(activation.Tanh(prevCell))

	prevHidden, err = gorgonia.BroadcastHadamardProd(outputGate, cellTan, nil, []byte{0})
	if err != nil {
		return nil, nil, err
	}

	return prevHidden, prevCell, nil
}
