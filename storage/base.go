package storage

import (
	"bytes"
	"encoding/gob"
	"errors"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	ErrLearnableNotFound = errors.New("learnable not found")
)

type Weight struct {
	Name  string
	Value gorgonia.Value
}

// GobEncode implements the gob.GobEncoder interface
func (w *Weight) GobEncode() ([]byte, error) {
	buf := new(bytes.Buffer)
	encoder := gob.NewEncoder(buf)

	err := encoder.Encode(&w.Name)
	if err != nil {
		return nil, err
	}

	err = encoder.Encode(&w.Value)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), err
}

// GobDecode implements the gob.GobDecoder interface
func (w *Weight) GobDecode(buf []byte) error {
	reader := bytes.NewBuffer(buf)
	decoder := gob.NewDecoder(reader)

	err := decoder.Decode(&w.Name)
	if err != nil {
		return err
	}

	err = decoder.Decode(&w.Value)
	if err != nil {
		return err
	}

	return nil
}

type Item struct {
	Cost    float32
	Weights []Weight
}

// NodesToItem converts a list of nodes to an storage.Item
func NodesToItem(nodes ...*gorgonia.Node) Item {
	item := Item{}
	item.Weights = make([]Weight, len(nodes))

	for i, n := range nodes {
		item.Weights[i].Name = n.Name()
		item.Weights[i].Value = n.Value().(tensor.Tensor)
	}

	return item
}
