package mind_gaze_gui;

import java.awt.*;
import java.io.File;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.ActionEvent;

public class GuiController {
	public static void main(String[] args) {
		Boolean analyze = false;
		FilePicker filePicker = new FilePicker();

		// Creating new Frame
		JFrame frame = new JFrame("MIND Gaze");

		// Screen dimensions
		Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		int screenWidth = screenSize.width;
		int screenHeight = screenSize.height;

		// Creating new container
		Container container = frame.getContentPane();
		container.setLayout(new BorderLayout(0, 0));

		// Panel at top for button
		JPanel topPanel = new JPanel();
		frame.getContentPane().add(topPanel, BorderLayout.NORTH);

		// Panel at center for video playback
		JPanel centerPanel = new JPanel();
		frame.getContentPane().add(centerPanel, BorderLayout.CENTER);

		// Panel at bottom for button
		JPanel bottomPannel = new JPanel();
		frame.getContentPane().add(bottomPannel, BorderLayout.SOUTH);

		// Adding import Button
		JButton importButton = new JButton("Import Video File");
		topPanel.add(importButton);
		importButton.setForeground(Color.BLACK);
		importButton.setToolTipText("Click on the button to Import a video file");

		// Adding analyze Button
		JButton analyzeButton = new JButton("Analyze Video!");
		bottomPannel.add(analyzeButton);
		importButton.setForeground(Color.BLACK);
		importButton.setToolTipText("Click on the button to analyze  the video file");

		// Import button Action - Picking a file
		importButton.addActionListener(filePicker);
		String filePath = filePicker.getFilePath();
		File file = filePicker.getFile();

		// TODO: Ability to Play Video

		// TODO: Analyze Button Action - turn a flag

		// Rendering the frame
		frame.setSize(screenWidth, screenHeight); // will cover the entire screen
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

	}

}
