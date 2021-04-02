package vggface2

import (
	"fmt"

	"github.com/dcu/godl"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type BlockOpts struct {
	Filters    [3]int
	Stride     []int
	Stage      int
	KernelSize tensor.Shape
	Block      int
}

func IdentityBlock(m *godl.Model, opts BlockOpts) godl.Layer {
	lt := godl.AddLayer("vggface2.IdentityBlock")

	conv1ReduceName := fmt.Sprintf("conv%d_%d_1x1_reduce", opts.Stage, opts.Block)
	conv1IncreaseName := fmt.Sprintf("conv%d_%d_1x1_increase", opts.Stage, opts.Block)
	conv3Name := fmt.Sprintf("conv%d_%d_3x3", opts.Stage, opts.Block)

	bn1 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[0],
		ScaleName: conv1ReduceName + "/bn/gamma",
		BiasName:  conv1ReduceName + "/bn/beta",
	})
	bn2 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[1],
		ScaleName: conv3Name + "/bn/gamma",
		BiasName:  conv3Name + "/bn/beta",
	})
	bn3 := godl.BatchNorm2d(m, godl.BatchNormOpts{
		InputSize: opts.Filters[2],
		ScaleName: conv1IncreaseName + "/bn/gamma",
		BiasName:  conv1IncreaseName + "/bn/beta",
	})

	return func(inputs ...*gorgonia.Node) (godl.Result, error) {
		if err := m.CheckArity(lt, inputs, 1); err != nil {
			return godl.Result{}, err
		}

		x := inputs[0]
		{
			w1 := m.AddWeights(lt, tensor.Shape{opts.Filters[0], 3, 3, 3}, godl.NewWeightsOpts{
				UniqueName: conv1ReduceName + "/kernel",
			})
			x = gorgonia.Must(gorgonia.Conv2d(x, w1, tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}))

			result, err := bn1(x)
			if err != nil {
				panic(err)
			}

			x = gorgonia.Must(gorgonia.Rectify(result.Output))
		}

		{
			w2 := m.AddWeights(lt, tensor.Shape{opts.Filters[1], opts.Filters[0], 3, 3}, godl.NewWeightsOpts{
				UniqueName: conv3Name + "/kernel",
			})
			x = gorgonia.Must(gorgonia.Conv2d(x, w2, opts.KernelSize, []int{0, 0}, []int{1, 1}, []int{1, 1}))

			result, err := bn2(x)
			if err != nil {
				panic(err)
			}

			x = gorgonia.Must(gorgonia.Rectify(result.Output))
		}

		{
			w3 := m.AddWeights(lt, tensor.Shape{opts.Filters[2], opts.Filters[1], 3, 3}, godl.NewWeightsOpts{
				UniqueName: conv1IncreaseName + "/kernel",
			})
			x = gorgonia.Must(gorgonia.Conv2d(x, w3, tensor.Shape{1, 1}, []int{0, 0}, []int{1, 1}, []int{1, 1}))

			result, err := bn3(x)
			if err != nil {
				panic(err)
			}

			x = gorgonia.Must(gorgonia.Add(result.Output, inputs[0]))
			x = gorgonia.Must(gorgonia.Rectify(x))
		}

		return godl.Result{Output: x}, nil
	}
}
