package godl

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
		input              tensor.Tensor
		expectedOutput     tensor.Tensor
		expectedOutputGrad tensor.Tensor
		expectedScaleGrad  tensor.Tensor
		expectedBiasGrad   tensor.Tensor
		expectedCost       float64
	}{
		{
			desc:               "Example 1",
			input:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.3, 0.03, 0.07, 0.7})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.9996221424388056, -0.9999554496246411, -0.9996221424388058, 0.999955449624641})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0.2500, 0.2500, 0.2500, 0.2500})),
			expectedScaleGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{0, 0})),
			expectedBiasGrad:   tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{0.5, 0.5})),
			expectedCost:       float64(-5.551115123125783e-17),
		},
		{
			desc:               "Example 2",
			input:              tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float64{0.3, 0.03, 0.07, 0.7})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float64{0.9996221424388056, -0.9999554496246411, -0.9996221424388058, 0.999955449624641})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float64{0.2500, 0.2500, 0.2500, 0.2500})),
			expectedScaleGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{0, 0})),
			expectedBiasGrad:   tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float64{0.5, 0.5})),
			expectedCost:       float64(-5.551115123125783e-17),
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.1))

			m := NewModel()
			m.Training = true

			bnFunc := BatchNorm1d
			if tC.input.Dims() == 4 {
				bnFunc = BatchNorm2d
			}

			bn := bnFunc(m, BatchNormOpts{
				InputSize: tC.input.Shape()[1],
			})

			x := gorgonia.NewTensor(m.g, tensor.Float64, tC.input.Shape().Dims(), gorgonia.WithShape(tC.input.Shape()...), gorgonia.WithValue(tC.input), gorgonia.WithName("x"))

			result, err := bn(x)
			c.NoError(err)

			cost := gorgonia.Must(gorgonia.Mean(result.Output))

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

			scaleGrad, err := l[0].Grad()
			c.NoError(err)

			biasGrad, err := l[1].Grad()
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
			c.Equal(tC.expectedCost, cost.Value().Data())

			c.NoError(solver.Step(gorgonia.NodesToValueGrads(m.Learnables())))

			log.Printf("scale: %v", l[0].Value())
			log.Printf("bias: %v", l[1].Value())
		})
	}
}
