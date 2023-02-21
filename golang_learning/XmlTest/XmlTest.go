package XmlTest

import (
	"encoding/xml"
	"fmt"
	"log"
	"os"
)

type Person struct {
	XMLName xml.Name `xml:"person"`
	Name    string   `xml:"name"`
	Age     int      `xml:"age"`
	Email   string   `xml:"email"`
}
type Temp struct {
	XMLName xml.Name `xml:"note"`
	To      string   `xml:"to"`
	From    string   `xml:"from"`
	Heading string   `xml:"heading"`
	Body    string   `xml:"body"`
}

func XmlMarshal() {
	p := Person{
		Name:  "Alexander",
		Age:   20,
		Email: "Alex@gmail.com",
	}
	b, err := xml.Marshal(p)
	if err != nil {
		log.Panic("Something wrong...")
	}
	fmt.Printf("b: %v\n", string(b))
}
func XmlMarshalIndent() {
	p := Person{
		Name:  "Alexander",
		Age:   20,
		Email: "Alex@gmail.com",
	}
	b, err := xml.MarshalIndent(p, "", "	")
	if err != nil {
		log.Panic("Something wrong...")
	}
	fmt.Printf("b: %v\n", string(b))
}
func XmlUnMarshal() {
	xml_string := `
	<person>
		<name>Alexander</name>
		<age>20</age>
		<email>Alex@gmail.com</email>
	</person>
	`
	b := []byte(xml_string)
	var p Person
	xml.Unmarshal(b, &p)
	fmt.Println(p)
}
func XmlReadfile() {
	f, err := os.OpenFile("test.xml", os.O_RDONLY, 0644)
	if err != nil {
		log.Panic("something wrong...")
	}
	defer f.Close()

	d := xml.NewDecoder(f)

	var v = Temp{}
	err2 := d.Decode(&v)
	if err2 != nil {
		log.Panic(err2)
	}
	fmt.Printf("v: %v\n", v)
}
func XmlWritefile() {
	f, err := os.OpenFile("test_write.xml", os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		log.Panic(err)
	}
	defer f.Close()

	e := xml.NewEncoder(f)
	e.Encode(Person{
		Name:  "Jackwolf Skin",
		Age:   2000,
		Email: "none",
	})
}
