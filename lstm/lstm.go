package lstm

import (
	"fmt"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type LSTMOpts struct {
	InputDimension int
	HiddenSize     int
	// Layers         int
	// Bidirectional bool
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
}

func LSTM(m *godl.Model, opts LSTMOpts) godl.Layer {
	opts.setDefaults()
	lt := godl.AddLayer("LSTM")

	inputWeights := m.AddWeights(lt, tensor.Shape{1, opts.InputDimension, opts.HiddenSize * 4}, godl.NewWeightsOpts{
		InitFN: opts.WeightsInit,
	})
	hiddenWeights := m.AddWeights(lt, tensor.Shape{1, opts.HiddenSize, opts.HiddenSize * 4}, godl.NewWeightsOpts{
		InitFN: opts.WeightsInit,
	})

	bias := m.AddBias(lt, tensor.Shape{1, 1, opts.HiddenSize * 4}, godl.NewWeightsOpts{
		InitFN: opts.BiasInit,
	})

	dummyHidden := gorgonia.NewMatrix(m.ExprGraph(), tensor.Float32, gorgonia.WithShape(1, opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyHidden"))
	dummyCell := gorgonia.NewMatrix(m.ExprGraph(), tensor.Float32, gorgonia.WithShape(1, opts.HiddenSize), gorgonia.WithInit(gorgonia.Zeroes()), gorgonia.WithName("LSTMDummyCell"))

	return func(inputs ...*gorgonia.Node) (godl.Result, error) {
		var (
			x, prevHidden, prevCell *gorgonia.Node
		)

		switch len(inputs) {
		case 1:
			x = inputs[0]
			prevHidden = dummyHidden
			prevCell = dummyCell
		case 3:
			x = inputs[0]
			prevHidden = inputs[1]
			prevCell = inputs[2]
		default:
			return godl.Result{}, fmt.Errorf("%v: invalid input size", lt)
		}

		seqs := x.Shape()[0]
		outputs := make([]*gorgonia.Node, seqs)

		for seq := 0; seq < seqs; seq++ {
			seqX := gorgonia.Must(gorgonia.Slice(x, gorgonia.S(seq), nil, nil))

			seqX = gorgonia.Must(gorgonia.Reshape(seqX, tensor.Shape{1, seqX.Shape()[0], seqX.Shape()[1]}))
			seqX = gorgonia.Must(gorgonia.BatchedMatMul(seqX, inputWeights))
			prevHidden = gorgonia.Must(gorgonia.BatchedMatMul(prevHidden, hiddenWeights))

			gates := gorgonia.Must(gorgonia.Add(seqX, prevHidden))
			gates = gorgonia.Must(gorgonia.BroadcastAdd(gates, bias, nil, []byte{0, 1}))

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
				return godl.Result{}, err
			}

			write, err := gorgonia.BroadcastHadamardProd(inputGate, cellGate, nil, []byte{0})
			if err != nil {
				return godl.Result{}, err
			}

			prevCell, err = gorgonia.Add(retain, write)
			if err != nil {
				return godl.Result{}, err
			}

			cellTan := gorgonia.Must(godl.Tanh(prevCell))

			prevHidden, err = gorgonia.BroadcastHadamardProd(outputGate, cellTan, nil, []byte{0})
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
}
