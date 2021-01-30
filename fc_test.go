package tabnet

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestFC(t *testing.T) {
	testCases := []struct {
		desc               string
		y                  tensor.Tensor
		input              tensor.Tensor
		expectedOutput     tensor.Tensor
		expectedOutputGrad tensor.Tensor
		expectedWeightGrad tensor.Tensor
	}{
		{
			desc:               "Example 1",
			y:                  tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{0.5, 0.05, 0.1, 0.2, 0.05, 0.5, 0.1, 0.2})),
			input:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.1, 0.01, 0.01, 0.1})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125})),
			expectedWeightGrad: tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375, 0.01375})),
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			m := NewModel()
			fc := FC(m, FCOpts{
				InputDimension:  tC.input.Shape()[1],
				OutputDimension: 4,
				WeightsInit:     gorgonia.Ones(),
			})

			x := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithShape(tC.input.Shape()...), gorgonia.WithValue(tC.input), gorgonia.WithName("x"))
			y := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithShape(tC.y.Shape()...), gorgonia.WithValue(tC.y), gorgonia.WithName("y"))

			result, err := fc(x)
			c.NoError(err)

			diff := gorgonia.Must(gorgonia.Sub(result.Output, y))
			cost := gorgonia.Must(gorgonia.Mean(diff))

			l := m.learnables

			_, err = gorgonia.Grad(cost, l...)
			c.NoError(err)

			_ = ioutil.WriteFile("fc.dot", []byte(m.g.ToDot()), 0644)

			vm := gorgonia.NewTapeMachine(m.g,
				gorgonia.BindDualValues(l...),
				gorgonia.WithWatchlist(),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())
			c.NoError(vm.Close())

			c.Equal(tC.expectedOutput.Data(), result.Output.Value().Data())

			outputGrad, err := result.Output.Grad()
			c.NoError(err)
			c.Equal(tC.expectedOutputGrad.Data(), outputGrad.Data())

			weightGrad, err := m.learnables[0].Grad()
			c.NoError(err)
			c.Equal(tC.expectedWeightGrad.Data(), weightGrad.Data())
		})
	}
}
