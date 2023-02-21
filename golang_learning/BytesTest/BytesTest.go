package BytesTest

import (
	"bytes"
	"fmt"
	"io"
)

func BytesContains() {
	s := "Visual Studio Code"
	b := []byte(s)
	b1 := []byte("VISUAL STUDIO CODE")
	b2 := []byte("Visual Studio Code")

	fmt.Printf("bytes.Contains(b, b2): %v\n", bytes.Contains(b, b2))
	fmt.Printf("bytes.Contains(b, b1): %v\n", bytes.Contains(b, b1))
	fmt.Printf("b1: %v\n", b1)
}
func Reader_Test() {
	var data string = "123aaaHHHgeek"
	b := []byte(data)
	r := bytes.NewReader(b)
	buf := make([]byte, 2)
	for {
		n, err := r.Read(buf)
		if err != nil {
			break
		}
		fmt.Println(string(buf[:n]))
	}

	r.Seek(0, 0) // back to beginning
	for {
		b2, err := r.ReadByte() //read one byte
		if err == io.EOF {
			break
		}
		fmt.Println(string(b2))
	}

	r.Seek(0, 0)
	offset := int64(0)
	for {
		n, err := r.ReadAt(buf, offset)
		if err == io.EOF {
			break
		}
		fmt.Println(string(buf[:n]))
		offset += int64(n)
	}
}
func BufferTest() {
	var buf bytes.Buffer      // it can directly use.
	buf2 := new(bytes.Buffer) //using new to create a buf
	buf3 := bytes.NewBuffer([]byte("using []byte slice"))
	buf4 := bytes.NewBufferString("or using this function")

	buf.Write([]byte("write to the end of buf"))
	buf2.WriteByte(byte('G')) //write one byte to buf2
	buf3.WriteString("write string to buf")
	buf4.WriteRune(int32(20013)) // ‘中’的UTF8编码，16进制转10进制转int32类型

	fmt.Printf("buf: %v\n", buf)
	fmt.Printf("buf2: %v\n", buf2)
	fmt.Printf("buf3: %v\n", buf3)
	fmt.Printf("buf4: %v\n", buf4)

}
