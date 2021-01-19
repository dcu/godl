package tabnet

import (
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func initDummyWeights(dt tensor.Dtype, s ...int) interface{} {
	v := make([]float64, tensor.Shape(s).TotalSize())

	for i := range v {
		v[i] = 1.0
	}

	return v
}

func TestGLU(t *testing.T) {
	testCases := []struct {
		desc           string
		input          tensor.Tensor
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float64{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{6, 5},
			expectedOutput: []float64{-0.26894085629326375, -0.26894085629326375, -0.26894085629326375, -0.26894085629326375, -0.26894085629326375, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574},
		},
		{
			desc: "Example 2",
			input: tensor.New(
				tensor.WithShape(6, 2),
				tensor.WithBacking([]float64{0.1, -0.5, 0.3, 0.9, 0.04, -0.3, 0.01, 0.09, -0.1, 0.9, 0.7, 0.04}),
			),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{6, 5},
			expectedOutput: []float64{-0.26894085629326375, -0.26894085629326375, -0.26894085629326375, -0.26894085629326375, -0.26894085629326375, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, 0.7310513312982878, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, -0.26893025839613094, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.73091545632948, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, 0.7259520054251364, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574, -0.2685383107725574},
		},
	}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)

			g := NewGraph()
			tn := &Model{g: g}

			input := NewTensor(g, tensor.Float64, tcase.input.Shape().Dims(), WithShape(tcase.input.Shape()...), WithName("input"), WithValue(tcase.input))

			x, _, err := tn.GLU(GLUOpts{
				InputDimension: tcase.vbs,
				OutputDimension:  tcase.output,
				WeightsInit:      initDummyWeights,
			})(input)

			if tcase.expectedErr != "" {
				c.Error(err)

				c.Equal(tcase.expectedErr, err.Error())

				return
			} else {
				c.NoError(err)
			}

			vm := NewTapeMachine(g)
			c.NoError(vm.RunAll())

			c.Equal(tcase.expectedShape, x.Shape())
			c.Equal(tcase.expectedOutput, x.Value().Data().([]float64))
		})
	}
}
