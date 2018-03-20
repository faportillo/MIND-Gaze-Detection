package mind_gaze_gui;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;

import javax.swing.*;
import javax.swing.filechooser.FileNameExtensionFilter;

import com.sun.jna.NativeLibrary;

import lombok.Data;
import uk.co.caprica.vlcj.component.DirectMediaPlayerComponent;
import uk.co.caprica.vlcj.player.MediaPlayerFactory;
import uk.co.caprica.vlcj.player.embedded.EmbeddedMediaPlayer;
import uk.co.caprica.vlcj.player.embedded.videosurface.CanvasVideoSurface;
import uk.co.caprica.vlcj.runtime.RuntimeUtil;

@Data
public class GuiController {
	// TODO: Need to put path of your VLC library and plugins here
	private static final String NATIVE_LIBRARY_SEARCH_PATH = "/Applications/VLC.app/Contents/MacOS/lib";
	private static String  SELECTED_FILE_PATH = "null";
	public static void main(String[] args) {
		
		NativeLibrary.addSearchPath(RuntimeUtil.getLibVlcLibraryName(), NATIVE_LIBRARY_SEARCH_PATH);

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
		centerPanel.setBorder(null);
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

		// vlcj only works with canvas
		 Canvas playerCanvas = new Canvas();
		 centerPanel.add(playerCanvas);
		 MediaPlayerFactory mediaPlayerFactory = new MediaPlayerFactory();
		 CanvasVideoSurface videoSurface =
		 mediaPlayerFactory.newVideoSurface(playerCanvas);
		 EmbeddedMediaPlayer mediaPlayer =
		 mediaPlayerFactory.newEmbeddedMediaPlayer();
		 mediaPlayer.setVideoSurface(videoSurface);

		// Import button Action - Picking a file
		importButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				JFileChooser file = new JFileChooser();
				FileNameExtensionFilter filter = new FileNameExtensionFilter("MOV, MPG & MP4 Videos", "mov", "mpg",
						"mp4");
				file.setFileFilter(filter);
				int result = file.showOpenDialog(null);
				if (result == JFileChooser.APPROVE_OPTION) {
					File selectedFile = file.getSelectedFile();
					final String filePath = selectedFile.getAbsolutePath();
					SELECTED_FILE_PATH = filePath;
					System.out.println("You chose to open this file: " + SELECTED_FILE_PATH);
					mediaPlayer.playMedia(filePath);					
				}
			}
		});

		// Analyze button Action - Kickstart Analysis
		analyzeButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// TODO: Python Script gets called here
				// SELECTED_FILE_PATH String has the selected file path that you need 
				String[] cmd={"python", "/home/felix/Documents/ML_Progs/MIND-Gaze-Detection",
					SELECTED_FILE_PATH,};
				Runtime.getRuntime().exec(cmd);
			}});

		// Rendering the frame
		frame.setSize(screenWidth, screenHeight); // will cover the entire screen
		frame.setVisible(true);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}

}