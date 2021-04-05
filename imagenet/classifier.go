package imagenet

import (
	"image"
	"sync"

	"github.com/dcu/godl"
	"github.com/dcu/godl/imageutils"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Classifier is an imagenet classifier
type Classifier struct {
	m *godl.Model
	x *gorgonia.Node

	output gorgonia.Value
	mutex  sync.Mutex
}

func NewClassifier(builder func(m *godl.Model) godl.Layer, width, height int) *Classifier {
	m := godl.NewModel()
	layer := builder(m)
	x := gorgonia.NewTensor(m.ExprGraph(), tensor.Float32, 4, gorgonia.WithShape(1, 3, width, height), gorgonia.WithName("x"))

	result, err := layer(x)
	if err != nil {
		panic(err)
	}

	c := &Classifier{
		m: m,
		x: x,
	}

	gorgonia.Read(result.Output, &c.output)

	return c
}

func (c *Classifier) Model() *godl.Model {
	return c.m
}

func (c *Classifier) Predict(img image.Image) (string, float32, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	input := imageutils.ToTensor(img, imageutils.ToTensorOpts{})

	err := gorgonia.Let(c.x, input)
	if err != nil {
		return "", 0.0, err
	}

	err = c.m.Run()
	if err != nil {
		return "", 0.0, err
	}

	outputTensor := c.output.(tensor.Tensor)
	max, err := tensor.Argmax(outputTensor, 1)
	if err != nil {
		return "", 0.0, err
	}

	index := max.Data().([]int)[0]

	val, err := outputTensor.At(0, index)
	if err != nil {
		return "", 0.0, err
	}

	return Labels[index], val.(float32), nil
}
