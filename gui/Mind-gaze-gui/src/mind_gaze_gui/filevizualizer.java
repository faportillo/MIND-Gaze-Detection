package mind_gaze_gui;

import java.awt.*;
import java.awt.event.MouseEvent;

import javax.swing.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;

public class filevizualizer {
	public static void main(String[] args) {
	
	// Creating new Frame	
    JFrame frame = new JFrame("MIND Gaze");

	// Screen dimensions
	Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
	int screenWidth = screenSize.width;
	int screenHeight = screenSize.height;
	
    // Creating new container
	Container container = frame.getContentPane();
	
	
	//Rendering the frame
	frame.setSize( screenWidth/2, screenHeight/2); //will cover the entire screen
	frame.setVisible(true);
	frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
}
	
}
