package godl

import (
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestFC(t *testing.T) {
	testCases := []struct {
		desc               string
		input              tensor.Tensor
		expectedOutput     tensor.Tensor
		expectedOutputGrad tensor.Tensor
		expectedInputGrad  tensor.Tensor
		expectedWeightGrad tensor.Tensor
	}{
		{
			desc:               "Example 1",
			input:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.1, 0.01, 0.01, 0.1})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{-0.0052000005, 0.0013999998, 0.008, 0.014599999, -0.0025000002, 0.0041, 0.0107, 0.0173})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 4), tensor.WithBacking([]float32{0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125})),
			expectedInputGrad:  tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.019999999999999997, 0.034999999999999996, 0.019999999999999997, 0.034999999999999996})),
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
				WeightsInit:     gorgonia.RangedFromWithStep(-0.05, 0.03),
			})

			x := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithShape(tC.input.Shape()...), gorgonia.WithValue(tC.input), gorgonia.WithName("x"))

			result, err := fc(x)
			c.NoError(err)

			cost := gorgonia.Must(gorgonia.Mean(result.Output))

			l := m.learnables

			_, err = gorgonia.Grad(cost, append(l, x)...)
			c.NoError(err)

			// _ = ioutil.WriteFile("fc.dot", []byte(m.g.ToDot()), 0644)

			vm := gorgonia.NewTapeMachine(m.g,
				gorgonia.BindDualValues(l...),
				gorgonia.WithWatchlist(),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())
			c.NoError(vm.Close())

			c.Equal(tC.expectedInputGrad.Data(), x.Deriv().Value().Data())
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
