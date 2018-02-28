package mind_gaze_gui;

import java.awt.*;
import java.awt.event.MouseEvent;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;

public class filechooser {
	public static void main(String[] args) {
		
		// Creating new Frame	
	    JFrame frame = new JFrame("MIND Gaze");

		// Screen dimensions
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int screenWidth = screenSize.width;
		int screenHeight = screenSize.height;
		
	    // Creating new container
		Container container = frame.getContentPane();
		
		// Adding Button
		JButton importButton = new JButton("Import Video File");
		importButton.setForeground(Color.BLACK);
		importButton.setToolTipText("Click on the button to Import a video file");
		//TODO: Change Button Size
	
		// Button Action - Picking a file
		importButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JFileChooser file = new JFileChooser();
				FileNameExtensionFilter filter = new FileNameExtensionFilter(
				        "MOV, MPG & MP4 Videos", "mov", "mpg", "mp4");
					file.setFileFilter(filter);
				    int returnVal = file.showOpenDialog(container);
				    if(returnVal == JFileChooser.APPROVE_OPTION) {
				       System.out.println("You chose to open this file: " +
				    		   file.getSelectedFile().getName());
			}
		}});
		
		container.setLayout(new BorderLayout(0, 0));
		container.add(importButton);
		
		// Panel on all ends just to make things visually better
		JPanel rightPanel = new JPanel();
		frame.getContentPane().add(rightPanel, BorderLayout.EAST);
		rightPanel.setSize(new Dimension(100, 100));
		
		JPanel leftPanel = new JPanel();
		frame.getContentPane().add(leftPanel, BorderLayout.WEST);
		
		JPanel topPanel = new JPanel();
		frame.getContentPane().add(topPanel, BorderLayout.NORTH);
		
		JPanel bottomPanel = new JPanel();
		frame.getContentPane().add(bottomPanel, BorderLayout.SOUTH);
		
		//Rendering the frame
		frame.setSize( screenWidth/2, screenHeight/2); //will cover the entire screen
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
		
}
