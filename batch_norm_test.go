package deepzen

import (
	"log"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestBatchNorm(t *testing.T) {
	testCases := []struct {
		desc               string
		y                  tensor.Tensor
		input              tensor.Tensor
		expectedOutput     tensor.Tensor
		expectedOutputGrad tensor.Tensor
		expectedScaleGrad  tensor.Tensor
		expectedBiasGrad   tensor.Tensor
	}{
		{
			desc:               "Example 1",
			y:                  tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.5, 0.05, 0.05, 0.5})),
			input:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.3, 0.03, 0.07, 0.7})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.99962217, -0.9999554, -0.9996221, 0.99995553})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.2500, 0.2500, 0.2500, 0.2500})),
			// expectedScaleGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{1.6191e-08, 2.2240e-08})), // TODO: pytorch/tensorflow BN version
			expectedScaleGrad: tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{1.4901161e-08, 2.9802322e-08})),
			expectedBiasGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{0.5, 0.5})),
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			m := NewModel()
			m.Training = true

			bn := BatchNorm(m, BatchNormOpts{
				InputDimension: tC.input.Shape()[1],
			})

			x := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithShape(tC.input.Shape()...), gorgonia.WithValue(tC.input), gorgonia.WithName("x"))
			y := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithShape(tC.y.Shape()...), gorgonia.WithValue(tC.y), gorgonia.WithName("y"))

			result, err := bn(x)
			c.NoError(err)

			diff := gorgonia.Must(gorgonia.Sub(result.Output, y))
			cost := gorgonia.Must(gorgonia.Mean(diff))

			l := m.learnables

			_, err = gorgonia.Grad(cost, l...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(m.g,
				gorgonia.BindDualValues(l...),
				gorgonia.WithWatchlist(),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())
			c.NoError(vm.Close())

			outputGrad, err := result.Output.Grad()
			c.NoError(err)

			scaleGrad, err := m.learnables[0].Grad()
			c.NoError(err)

			biasGrad, err := m.learnables[1].Grad()
			c.NoError(err)

			log.Printf("input: %v", tC.input)
			log.Printf("output: %v", result.Value())
			log.Printf("output grad: %v", outputGrad)
			log.Printf("scale grad: %v", scaleGrad)
			log.Printf("bias grad: %v", biasGrad)
			log.Printf("cost: %v", cost.Value())

			c.Equal(tC.expectedOutput.Data(), result.Value().Data())
			c.Equal(tC.expectedOutputGrad.Data(), outputGrad.Data())
			c.Equal(tC.expectedScaleGrad.Data(), scaleGrad.Data())
			c.Equal(tC.expectedBiasGrad.Data(), biasGrad.Data())
		})
	}
}
