package lstm

import (
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestLSTM(t *testing.T) {
	testCases := []struct {
		desc        string
		Input       tensor.Tensor
		HiddenInput tensor.Tensor
		CellInput   tensor.Tensor

		HiddenSize    int
		Bidirectional bool
		WithBias      bool

		ExpectedOutput       tensor.Tensor
		ExpectedHiddenOutput tensor.Tensor
		ExpectedCellOutput   tensor.Tensor
	}{
		{
			desc:       "Example 1",
			WithBias:   true,
			HiddenSize: 3,
			Input: tensor.New(
				tensor.WithShape(1, 1, 2),
				tensor.WithBacking([]float32{0.2653, 1.0994}),
			),
			HiddenInput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{1.0118, -0.5075, 1.6139}),
			),
			CellInput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{0.0478, 0.1320, 2.6778}),
			),
			ExpectedOutput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{0.74483687, 0.77539575, 0.9686491}),
			),
			ExpectedHiddenOutput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{0.74483687, 0.77539575, 0.9686491}),
			),
			ExpectedCellOutput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{1.0147436, 1.0964341, 3.5663624}),
			),
		},
		{
			desc:       "Example 2",
			HiddenSize: 3,
			WithBias:   true,
			Input: tensor.New(
				tensor.WithShape(2, 1, 2),
				tensor.WithBacking([]float32{1.5333, -2.5040, -0.9077, -0.2410}),
			),
			HiddenInput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{-1.4728, -0.1176, -0.6555}),
			),
			CellInput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{-0.1723, -0.7119, 0.9053}),
			),
			ExpectedOutput: tensor.New(
				tensor.WithShape(2, 1, 3),
				tensor.WithBacking([]float32{-0.0017358345, -0.0025350708, -0.00013593411, -0.049025163, -0.050171264, -0.046729423}),
			),
			ExpectedHiddenOutput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{-0.049025163, -0.050171264, -0.046729423}),
			),
			ExpectedCellOutput: tensor.New(
				tensor.WithShape(1, 1, 3),
				tensor.WithBacking([]float32{-0.20725529, -0.21224551, -0.19728966}),
			),
		},
		{
			desc:       "Example 3",
			HiddenSize: 3,
			WithBias:   true,
			Input: tensor.New(
				tensor.WithShape(2, 2, 2),
				tensor.WithBacking([]float32{1.0138, -1.3644, 0.7197, 0.6099, 0.6193, 1.8894, -0.8197, 1.6288}),
			),
			HiddenInput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{-0.9553, -2.0582, 1.7633, 0.8580, -1.6359, 0.4142}),
			),
			CellInput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{0.8895, -0.2665, -0.6879, 0.2842, 0.0325, -0.9686}),
			),
			ExpectedOutput: tensor.New(
				tensor.WithShape(2, 2, 3),
				tensor.WithBacking([]float32{-0.0009099007, -0.03304913, -0.04428876, 0.45866472, 0.37035143, -0.1153187, 0.65836966, 0.5679527, 0.5295214, 0.7193798, 0.6868317, 0.44933355}),
			),
			ExpectedHiddenOutput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{0.65836966, 0.5679527, 0.5295214, 0.7193798, 0.6868317, 0.44933355}),
			),
			ExpectedCellOutput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{0.90001315, 0.72165096, 0.65663207, 1.3595006, 1.209836, 0.61456656}),
			),
		},
		{
			desc:       "Example 4",
			HiddenSize: 3,
			WithBias:   false,
			Input: tensor.New(
				tensor.WithShape(2, 2, 2),
				tensor.WithBacking([]float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}),
			),
			HiddenInput: nil,
			CellInput:   nil,
			ExpectedOutput: tensor.New(
				tensor.WithShape(2, 2, 3),
				tensor.WithBacking([]float32{0.007723757, 0.007723757, 0.007723757, 0.018707205, 0.018707205, 0.018707205, 0.041886266, 0.041886266, 0.041886266, 0.07235941, 0.07235941, 0.07235941}),
			),
			ExpectedHiddenOutput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{0.041886266, 0.041886266, 0.041886266, 0.07235941, 0.07235941, 0.07235941}),
			),
			ExpectedCellOutput: tensor.New(
				tensor.WithShape(1, 2, 3),
				tensor.WithBacking([]float32{0.078712106, 0.078712106, 0.078712106, 0.13200212, 0.13200212, 0.13200212}),
			),
		},
		{
			desc:          "Example 5 (Bidirectional)",
			HiddenSize:    3,
			Bidirectional: true,
			WithBias:      false,
			Input: tensor.New(
				tensor.WithShape(2, 2, 2),
				tensor.WithBacking([]float32{0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08}),
			),
			HiddenInput: nil,
			CellInput:   nil,
			ExpectedOutput: tensor.New(
				tensor.WithShape(2, 2, 6),
				tensor.WithBacking([]float32{0.007723757, 0.007723757, 0.007723757, 0.05005947, 0.05005947, 0.05005947, 0.018707205, 0.018707205, 0.018707205, 0.082762234, 0.082762234, 0.082762234, 0.041886266, 0.041886266, 0.041886266, 0.030448243, 0.030448243, 0.030448243, 0.07235941, 0.07235941, 0.07235941, 0.042911056, 0.042911056, 0.042911056}),
			),
			ExpectedHiddenOutput: tensor.New(
				tensor.WithShape(2, 2, 3),
				tensor.WithBacking([]float32{0.041886266, 0.041886266, 0.041886266, 0.07235941, 0.07235941, 0.07235941, 0.05005947, 0.05005947, 0.05005947, 0.082762234, 0.082762234, 0.082762234}),
			),
			ExpectedCellOutput: tensor.New(
				tensor.WithShape(2, 2, 3),
				tensor.WithBacking([]float32{0.078712106, 0.078712106, 0.078712106, 0.13200212, 0.13200212, 0.13200212, 0.09468048, 0.09468048, 0.09468048, 0.1517626, 0.1517626, 0.1517626}),
			),
		},
	}

	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			m := godl.NewModel()
			input := gorgonia.NodeFromAny(m.ExprGraph(), tC.Input, gorgonia.WithName("input"))

			args := make([]*gorgonia.Node, 0, 3)
			args = append(args, input)

			if tC.HiddenInput != nil {
				hiddenInput := gorgonia.NodeFromAny(m.ExprGraph(), tC.HiddenInput, gorgonia.WithName("hidden"))
				args = append(args, hiddenInput)
			}

			if tC.CellInput != nil {
				cellInput := gorgonia.NodeFromAny(m.ExprGraph(), tC.CellInput, gorgonia.WithName("cell"))
				args = append(args, cellInput)
			}

			l := LSTM(m, LSTMOpts{
				InputDimension: input.Shape()[2],
				Bidirectional:  tC.Bidirectional,
				HiddenSize:     tC.HiddenSize,
				WeightsInit:    gorgonia.Ones(),
				BiasInit:       gorgonia.Zeroes(),
			})

			result, err := l(args...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(m.ExprGraph())
			c.NoError(vm.RunAll())

			m.PrintWatchables()

			// log.Printf("output: %v %v \n", result.Output.Shape(), result.Output.Value())
			// log.Printf("hidden output: %v %v \n", result.Nodes[0].Shape(), result.Nodes[0].Value())
			// log.Printf("cell output: %v %v \n", result.Nodes[1].Shape(), result.Nodes[1].Value())

			maxDiffAllowed := 1e-7

			c.InDeltaSlice(tC.ExpectedOutput.Data(), result.Output.Value().Data(), maxDiffAllowed)
			c.Equal(tC.ExpectedOutput.Shape(), result.Output.Shape())
			c.InDeltaSlice(tC.ExpectedHiddenOutput.Data(), result.Nodes[0].Value().Data(), maxDiffAllowed)
			c.Equal(tC.ExpectedHiddenOutput.Shape(), result.Nodes[0].Shape())
			c.InDeltaSlice(tC.ExpectedCellOutput.Data(), result.Nodes[1].Value().Data(), maxDiffAllowed)
			c.Equal(tC.ExpectedCellOutput.Shape(), result.Nodes[1].Shape())
		})
	}
}
