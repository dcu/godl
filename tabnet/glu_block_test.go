package tabnet

import (
	"fmt"
	"testing"

	"github.com/dcu/godl"
	"github.com/stretchr/testify/require"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestGLUBlock(t *testing.T) {
	testCases := []struct {
		X         tensor.Tensor
		BlockSize int
		Shared    int
		Output    int
		VBS       int
	}{
		{
			X: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float32{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			VBS:       3,
			Shared:    5,
			Output:    5,
			BlockSize: 2,
		},
	}
	for i, tC := range testCases {
		t.Run(fmt.Sprintf("#%d", i+1), func(t *testing.T) {
			c := require.New(t)
			nn := godl.NewModel()

			input := gorgonia.NewTensor(nn.ExprGraph(), tensor.Float32, tC.X.Dims(), gorgonia.WithShape(tC.X.Shape()...), gorgonia.WithName("x"), gorgonia.WithValue(tC.X))

			shared := make([]godl.Layer, tC.Shared)
			fcInput := input.Shape()[1]
			fcOutput := 2 * tC.Output
			for i := 0; i < tC.Shared; i++ {
				shared[i] = godl.FC(nn, godl.FCOpts{
					OutputDimension: fcOutput, // double the size so we can take half and half
					WeightsInit:     gorgonia.RangedFromWithStep(-0.1, 0.01),
					InputDimension:  fcInput,
				})

				fcInput = tC.Output
			}

			result, err := GLUBlock(nn, GLUBlockOpts{
				InputDimension:   tC.X.Shape()[1],
				OutputDimension:  tC.Output,
				Shared:           shared,
				VirtualBatchSize: tC.VBS,
				Size:             tC.BlockSize,
			})(input)
			c.NoError(err)

			y := result.Output
			cost := gorgonia.Must(gorgonia.Mean(y))
			_, err = gorgonia.Grad(cost, input)
			c.NoError(err)

			vm := gorgonia.NewTapeMachine(nn.ExprGraph(), gorgonia.BindDualValues(nn.Learnables()...))
			c.NoError(vm.RunAll())

			nn.PrintWatchables()

			t.Logf("y: %v", y.Value())
			t.Logf("dx: %v", input.Deriv().Value())
		})
	}
}
