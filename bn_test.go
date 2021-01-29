package tabnet

import (
	"io/ioutil"
	"log"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestBN(t *testing.T) {
	testCases := []struct {
		desc           string
		y              tensor.Tensor
		input          tensor.Tensor
		expectedOutput tensor.Tensor
	}{
		{
			desc:           "Example 1",
			y:              tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.5, 0.05, 0.05, 0.5})),
			input:          tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.1, 0.01, 0.01, 0.1})),
			expectedOutput: tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{0.99754, -0.99754, -0.99754, 0.99754})),
		},
	}
	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			c := require.New(t)

			m := NewModel()
			bn := BN(m, BNOpts{
				InputDimension: tC.input.Shape()[1],
			})

			x := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithValue(tC.input), gorgonia.WithName("x"))
			y := gorgonia.NewTensor(m.g, tensor.Float32, 2, gorgonia.WithValue(tC.y), gorgonia.WithName("y"))

			outputNode, _, err := bn(x)
			c.NoError(err)

			diff := gorgonia.Must(gorgonia.Sub(outputNode, y))
			cost := gorgonia.Must(gorgonia.Mean(diff))

			_, err = gorgonia.Grad(cost, m.learnables...)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(m.g,
				gorgonia.BindDualValues(m.learnables...),
				gorgonia.WithWatchlist(),
				gorgonia.TraceExec(),
			)
			c.NoError(vm.RunAll())
			c.NoError(vm.Close())

			c.Equal(tC.expectedOutput.Data(), outputNode.Value().Data())

			_ = ioutil.WriteFile("bn.dot", []byte(m.g.ToDot()), 0644)

			log.Printf("output: %v", outputNode.Value())
			log.Printf("cost: %v", cost.Value())
		})
	}
}
