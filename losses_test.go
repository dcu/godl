package godl

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestMSELoss(t *testing.T) {
	testCases := []struct {
		desc         string
		output       tensor.Tensor
		target       tensor.Tensor
		expectedLoss float64
	}{
		{
			desc: "Example 1",
			output: tensor.New(
				tensor.WithShape(1),
				tensor.WithBacking([]float64{0.5}),
			),
			target: tensor.New(
				tensor.WithShape(1),
				tensor.WithBacking([]float64{0.1}),
			),
			expectedLoss: 0.16000000000000003,
		},
		{
			desc: "Example 2",
			output: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float64{0.5, 0.2, 0.5, 0.7}),
			),
			target: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.9}),
			),
			expectedLoss: 0.06000000000000002,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()

			outputNode := gorgonia.NewTensor(g, tensor.Float64, tC.output.Shape().Dims(), gorgonia.WithShape(tC.output.Shape()...), gorgonia.WithValue(tC.output), gorgonia.WithName("output"))
			targetNode := gorgonia.NewTensor(g, tensor.Float64, tC.target.Shape().Dims(), gorgonia.WithShape(tC.target.Shape()...), gorgonia.WithValue(tC.target), gorgonia.WithName("target"))

			loss := MSELoss(outputNode, targetNode, MSELossOpts{})

			var lossV gorgonia.Value
			gorgonia.Read(loss, &lossV)

			vm := gorgonia.NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tC.expectedLoss, lossV.Data())
		})
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	testCases := []struct {
		desc         string
		reduction    Reduction
		output       tensor.Tensor
		target       tensor.Tensor
		expectedLoss float64
	}{
		{
			desc:      "Example 1",
			reduction: ReductionSum,
			output: tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float64{0.5, 0.1}),
			),
			target: tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float64{1, 0}),
			),
			expectedLoss: 0.6931471805599453,
		},
		{
			desc:      "Example 2",
			reduction: ReductionMean,
			output: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float64{0.5, 0.2, 0.5, 0.7}),
			),
			target: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float64{0.1, 0.2, 0.3, 0.9}),
			),
			expectedLoss: 0.23003847606391437,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()

			outputNode := gorgonia.NewTensor(g, tensor.Float64, tC.output.Shape().Dims(), gorgonia.WithShape(tC.output.Shape()...), gorgonia.WithValue(tC.output), gorgonia.WithName("output"))
			targetNode := gorgonia.NewTensor(g, tensor.Float64, tC.target.Shape().Dims(), gorgonia.WithShape(tC.target.Shape()...), gorgonia.WithValue(tC.target), gorgonia.WithName("target"))

			loss := CrossEntropyLoss(outputNode, targetNode, CrossEntropyLossOpt{
				Reduction: tC.reduction,
			})

			var lossV gorgonia.Value
			gorgonia.Read(loss, &lossV)

			vm := gorgonia.NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tC.expectedLoss, lossV.Data())
		})
	}
}
