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
		expectedLoss float32
	}{
		{
			desc: "Example 1",
			output: tensor.New(
				tensor.WithShape(1),
				tensor.WithBacking([]float32{0.5}),
			),
			target: tensor.New(
				tensor.WithShape(1),
				tensor.WithBacking([]float32{0.1}),
			),
			expectedLoss: 0.16000001,
		},
		{
			desc: "Example 2",
			output: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float32{0.5, 0.2, 0.5, 0.7}),
			),
			target: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float32{0.1, 0.2, 0.3, 0.9}),
			),
			expectedLoss: 0.06000000000000002,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()

			outputNode := gorgonia.NewTensor(g, tensor.Float32, tC.output.Shape().Dims(), gorgonia.WithShape(tC.output.Shape()...), gorgonia.WithValue(tC.output), gorgonia.WithName("output"))
			targetNode := gorgonia.NewTensor(g, tensor.Float32, tC.target.Shape().Dims(), gorgonia.WithShape(tC.target.Shape()...), gorgonia.WithValue(tC.target), gorgonia.WithName("target"))

			loss := MSELoss(MSELossOpts{})(Nodes{outputNode}, targetNode)

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
		expectedLoss float32
	}{
		{
			desc:      "Example 1",
			reduction: ReductionSum,
			output: tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float32{0.5, 0.1}),
			),
			target: tensor.New(
				tensor.WithShape(2),
				tensor.WithBacking([]float32{1, 0}),
			),
			expectedLoss: 0.6931471805599453,
		},
		{
			desc:      "Example 2",
			reduction: ReductionMean,
			output: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float32{0.5, 0.2, 0.5, 0.7}),
			),
			target: tensor.New(
				tensor.WithShape(2, 2),
				tensor.WithBacking([]float32{0.1, 0.2, 0.3, 0.9}),
			),
			expectedLoss: 0.2300385,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			g := gorgonia.NewGraph()

			outputNode := gorgonia.NewTensor(g, tensor.Float32, tC.output.Shape().Dims(), gorgonia.WithShape(tC.output.Shape()...), gorgonia.WithValue(tC.output), gorgonia.WithName("output"))
			targetNode := gorgonia.NewTensor(g, tensor.Float32, tC.target.Shape().Dims(), gorgonia.WithShape(tC.target.Shape()...), gorgonia.WithValue(tC.target), gorgonia.WithName("target"))

			loss := CrossEntropyLoss(CrossEntropyLossOpt{
				Reduction: tC.reduction,
			})(Nodes{outputNode}, targetNode)

			var lossV gorgonia.Value
			gorgonia.Read(loss, &lossV)

			vm := gorgonia.NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tC.expectedLoss, lossV.Data())
		})
	}
}
