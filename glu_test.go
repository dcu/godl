package tabnet

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func initDummyWeights(dt tensor.Dtype, s ...int) interface{} {
	v := make([]float64, tensor.Shape(s).TotalSize())
	rs := rand.New(rand.NewSource(0)) // fixed rand source, will generate same numbers in same order

	for i := range v {
		v[i] = rs.Float64()
	}

	return v
}

func TestGLU(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc           string
		input          *Node
		vbs            int
		output         int
		expectedShape  tensor.Shape
		expectedErr    string
		expectedOutput []float64
	}{
		{
			desc: "Example 1",
			input: NewTensor(g, tensor.Float64, 2, WithShape(10, 1), WithName("input"), WithValue(
				tensor.New(
					tensor.WithShape(10, 1),
					tensor.WithBacking([]float64{0.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4}),
				),
			)),
			vbs:            2,
			output:         5,
			expectedShape:  tensor.Shape{10, 5},
			expectedOutput: []float64{0.272798553560106, -0.2689874697718568, 0.7306916543389329, -0.7307601163293139, 0.26363991887305216, 0.26926397792553836, -0.2689451870078321, 0.7310286065016189, -0.7310342023375996, 0.2684964711188217, 0.2690513378893229, -0.2689427028643958, 0.7310483793988481, -0.7310502836793469, 0.2687897637017986, 0.26899620983582473, -0.26894205991703796, 0.7310534966109064, -0.7310544454748473, 0.2688658224809565, 0.268974140840554, -0.2689418026552361, 0.7310555441068347, -0.7310561106849224, 0.2688962729559755, 0.2689631461958882, -0.26894167451544576, 0.7310565639358336, -0.7310569401016058, 0.2689114437260297, 0.2689568882317996, -0.2689416015883443, 0.7310571443393163, -0.731057412137605, 0.26892007882419955, 0.26895299075113405, -0.2689415561719935, 0.731057505792408, -0.7310577061034533, 0.26892545684855895, 0.26895040025858424, -0.26894152598686916, 0.7310577460249376, -0.7310579014818932, 0.2689290314207389, 0.26894859156133055, -0.2689415049120171, 0.7310579137518677, -0.7310580378923125, 0.26893152721963326},
		},
	}

	tn := &Model{g: g}

	for _, tcase := range testCases {
		t.Run(tcase.desc, func(t *testing.T) {
			c := require.New(t)
			x, err := tn.GLU(GLUOpts{
				VirtualBatchSize: tcase.vbs,
				OutputFeatures:   tcase.output,
				WeightsInit:      initDummyWeights,
			})(tcase.input)

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
