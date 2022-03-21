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
		expectedCost       float32
	}{
		{
			desc:               "Example 1",
			input:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.3, 0.03, 0.07, 0.7})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.9996221424388056, -0.9999554496246411, -0.9996221424388058, 0.999955449624641})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.2500, 0.2500, 0.2500, 0.2500})),
			expectedScaleGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{2.9802322e-08, 2.9802322e-08})),
			expectedBiasGrad:   tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{0.5, 0.5})),
			expectedCost:       0,
		},
		{
			desc:               "Example 2",
			input:              tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float32{0.3, 0.03, 0.07, 0.7})),
			expectedOutput:     tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float32{0.9996221424388056, -0.9999554496246411, -0.9996221424388058, 0.999955449624641})),
			expectedOutputGrad: tensor.New(tensor.WithShape(2, 2, 1, 1), tensor.WithBacking([]float32{0.2500, 0.2500, 0.2500, 0.2500})),
			expectedScaleGrad:  tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{2.9802322e-08, 2.9802322e-08})),
			expectedBiasGrad:   tensor.New(tensor.WithShape(1, 2), tensor.WithBacking([]float32{0.5, 0.5})),
			expectedCost:       0,
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.1))

			m := NewModel()

			bnFunc := BatchNorm1d
			if tC.input.Dims() == 4 {
				bnFunc = BatchNorm2d
			}

			bn := bnFunc(m, BatchNormOpts{
				InputSize: tC.input.Shape()[1],
			})

			x := gorgonia.NewTensor(m.trainGraph, tensor.Float32, tC.input.Shape().Dims(), gorgonia.WithShape(tC.input.Shape()...), gorgonia.WithValue(tC.input), gorgonia.WithName("x"))

			result, err := bn(x)
			c.NoError(err)

			cost := gorgonia.Must(gorgonia.Mean(result.Output))

			l := m.learnables

			_, err = gorgonia.Grad(cost, l...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(m.trainGraph,
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

			c.InDeltaSlice(tC.expectedOutput.Data(), result.Value().Data(), 1e-5, "actual: %#v", result.Value().Data())
			c.Equal(tC.expectedOutputGrad.Data(), outputGrad.Data())
			c.Equal(tC.expectedScaleGrad.Data(), scaleGrad.Data())
			c.Equal(tC.expectedBiasGrad.Data(), biasGrad.Data())
			c.InDelta(tC.expectedCost, cost.Value().Data(), 1e-5)

			c.NoError(solver.Step(gorgonia.NodesToValueGrads(m.Learnables())))

			log.Printf("scale: %v", l[0].Value())
			log.Printf("bias: %v", l[1].Value())
		})
	}
}
