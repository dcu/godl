package lstm

import (
	"fmt"

	"github.com/dcu/godl"
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

	RecurrentActivation godl.ActivationFn
	Activation          godl.ActivationFn
}

func (o *LSTMOpts) setDefaults() {
	if o.RecurrentActivation == nil {
		o.RecurrentActivation = godl.Sigmoid
	}

	if o.Activation == nil {
		o.Activation = godl.Tanh
	}

	if o.MergeMode == "" {
		o.MergeMode = MergeModeConcat
	}
}

func LSTM(m *godl.Model, opts LSTMOpts) godl.Layer {
	opts.setDefaults()
	lt := godl.AddLayer("LSTM")

	paramsCount := 1
	if opts.Bidirectional {
		paramsCount = 2
	}

	weights := buildParamsList(paramsCount, m, lt, opts)
	two := gorgonia.NewConstant(float32(2.0), gorgonia.WithName("two"))

	return func(inputs ...*gorgonia.Node) (godl.Result, error) {
		var (
			x, prevHidden, prevCell *gorgonia.Node
		)

		switch len(inputs) {
		case 1:
			x = inputs[0]
			batchSize := x.Shape()[1]

			dummyHidden := gorgonia.NewTensor(m.ExprGraph(), tensor.Float32, 3, gorgonia.WithShape(1, batchSize, opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyHidden")) // FIXME: unique name
			dummyCell := gorgonia.NewTensor(m.ExprGraph(), tensor.Float32, 3, gorgonia.WithShape(1, batchSize, opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyCell"))

			prevHidden = dummyHidden
			prevCell = dummyCell
		case 3:
			x = inputs[0]
			prevHidden = inputs[1]
			prevCell = inputs[2]
		default:
			return godl.Result{}, fmt.Errorf("%v: invalid input size", lt)
		}

		if !opts.Bidirectional {
			return lstm(m, x, weights[0].inputWeights, prevHidden, weights[0].hiddenWeights, prevCell, weights[0].bias, false, opts)
		}

		x1 := gorgonia.Must(gorgonia.BatchedMatMul(x, weights[0].inputWeights))
		forwardOutput, err := lstm(m, x1, weights[0].inputWeights, prevHidden, weights[0].hiddenWeights, prevCell, weights[0].bias, true, opts)
		if err != nil {
			return godl.Result{}, err
		}

		x2 := gorgonia.Must(gorgonia.BatchedMatMul(x, weights[1].inputWeights))
		x2 = gorgonia.Must(gorgonia.ApplyOp(OrderingOp{}, x2))

		backwardOutput, err := lstm(m, x2, weights[1].inputWeights, prevHidden, weights[1].hiddenWeights, prevCell, weights[1].bias, true, opts)
		if err != nil {
			return godl.Result{}, err
		}

		backwardOutputReversed := gorgonia.Must(gorgonia.ApplyOp(OrderingOp{}, backwardOutput.Output))

		result := godl.Result{}

		switch opts.MergeMode {
		case MergeModeAverage:
			result.Output = gorgonia.Must(gorgonia.Div(gorgonia.Must(gorgonia.Add(forwardOutput.Output, backwardOutputReversed)), two))
		case MergeModeConcat:
			result.Output = gorgonia.Must(gorgonia.Concat(forwardOutput.Output.Dims()-1, forwardOutput.Output, backwardOutputReversed))
		case MergeModeMul:
			result.Output = gorgonia.Must(gorgonia.HadamardProd(forwardOutput.Output, backwardOutputReversed))
		case MergeModeSum:
			result.Output = gorgonia.Must(gorgonia.Add(forwardOutput.Output, backwardOutputReversed))
		}

		hidden := gorgonia.Must(gorgonia.Concat(0, forwardOutput.Nodes[0], backwardOutput.Nodes[0]))
		cell := gorgonia.Must(gorgonia.Concat(0, forwardOutput.Nodes[1], backwardOutput.Nodes[1]))
		result.Nodes = gorgonia.Nodes{hidden, cell}

		return result, nil
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

func lstm(m *godl.Model, x, inputWeights, prevHidden, hiddenWeights, prevCell, bias *gorgonia.Node, withPrecomputedInput bool, opts LSTMOpts) (godl.Result, error) {
	seqs := x.Shape()[0]
	outputs := make([]*gorgonia.Node, seqs)

	var err error

	for seq := 0; seq < seqs; seq++ {
		seqX := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(seq), nil, nil))
		seqX = gorgonia.Must(gorgonia.Reshape(seqX, tensor.Shape{1, seqX.Shape()[0], seqX.Shape()[1]}))

		prevHidden, prevCell, err = lstmGate(m, seqX, inputWeights, prevHidden, hiddenWeights, prevCell, bias, withPrecomputedInput, opts)
		if err != nil {
			return godl.Result{}, err
		}

		outputs[seq] = prevHidden
	}

	outputGate := gorgonia.Must(gorgonia.Concat(0, outputs...))

	return godl.Result{
		Output: outputGate,
		Nodes: gorgonia.Nodes{
			prevHidden,
			prevCell,
		},
	}, nil
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

	cellTan := gorgonia.Must(godl.Tanh(prevCell))

	prevHidden, err = gorgonia.BroadcastHadamardProd(outputGate, cellTan, nil, []byte{0})
	if err != nil {
		return nil, nil, err
	}

	return prevHidden, prevCell, nil
}
