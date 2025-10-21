# Handle connections and data between sg-control and gnuradio

import socket, subprocess, threading, argparse, logging, traceback, os, json


logging.basicConfig(level=logging.INFO, handlers=[])



class GRH:
	def __init__(self, sock_path, verbose = False, init_cmd = None, local = False):
		logging.info('Init process')
		self.sock_path = sock_path
		self.verbose = verbose
		self.local = local
		self.devices = {}  # port → subprocess
		self.server = None
		self.init_cmd = init_cmd

	def handle_command(self, line, conn):
		if self.verbose:
			logging.info(f"Command coming in: {line}")
		parts = line.strip().split()
		if not parts: return
		response = None
		action = parts[0]
		args = parts[1:]

		if self.verbose:
			logging.info(f"Command action: {action}")

		if action == "open" and len(args) >= 6:
			dev_type, port, device, samp_rate, target_rate, freq, gain, *additional_args = args
			if not additional_args:
				additional_args = ""
			response = self.spawn_device(dev_type, int(port), device, int(samp_rate), int(target_rate), float(freq), gain, additional_args, conn)

		elif action == "close" and args:
			port = int(args[0])
			response = self.kill_device(port)

		elif action in ( "rf_gain", "if_gain" ) and len(args) == 2:
			port, value = args
			try:
				self.devices[port].stdin.write(f"set_{action} {value}\n")
				self.devices[port].stdin.flush()
			finally:
				response = {"status": "command_sent", "port": port, "action": action, "value": value}

		elif action == "set_freq" and args:
			response = self.src.set_center_freq(float(args[0]))

		elif action == "start":
			response = self.tb.start()

		elif action == "stop":
			response = self.tb.stop()

		if response is not None:
			return json.dumps( response )
		else:
			return json.dumps( {"status": "error", "message": f"Unknown command {action}", "args": args} )

	def spawn_device(self, dev_type, port, device, samp_rate, target_rate, freq, gain, additional_args, conn):
		if port in self.devices:
			self.kill_device(port)

		script = f"/usr/bin/gr_{dev_type}.py"
		cmd = [
			"python3", script,
			"--port", str(port),
			"--device", str(device),
			"--freq", str(freq),
			"--samp_rate", str(samp_rate),
			"--target_rate", str(target_rate),
			"--gain", str(gain),
			"--additional_args", str(additional_args)
		]
		try:
			proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		except Exception as e:
			return {"status": "error", "message": str(e)}

		self.devices[port] = proc

		if conn is not None:
			threading.Thread(target=self.pipe_output, args=(proc.stdout,proc.stderr,conn), daemon=True).start()
		else:
			threading.Thread(target=self.log_output, args=(proc.stdout,proc.stderr), daemon=True).start()

		return {"status": f"Spawned device on port:{port}", "script": script, "cmd": cmd}

	def kill_device(self, port):
		proc = self.devices.pop(port, None)
		if proc:
			proc.terminate()
		return {"status":"terminated", "port": port}

	def pipe_output(self, stream, error, conn):
		for line in stream:
			message = {"status": "data", "data": line}
			conn.sendall( ( json.dumps( message ) + "\n").encode() )
			if (self.verbose):
				logging.info(line)
		error = error.read()
		if error:
			message = {"status": "error", "error": f"\"{error}\""}
			conn.sendall( ( json.dumps( message ) + "\n").encode() )
	
	def handle_session(self, conn):
		with conn:
			try:
				buffer = ""
				while True:
					chunk = conn.recv(1024).decode()
					if not chunk:
						break
					buffer += chunk
					while "\n" in buffer:
						line, buffer = buffer.split("\n", 1)
						response = self.handle_command(line.strip(), conn)
						if response:
							conn.sendall((response + "\n").encode())
							if self.verbose:
								logging.info(response)
			except Exception as e:
				logging.error(f"Session error: {e}")

		
	def log_output(self, stream, errors):
		for line in stream:
			logging.info(line)
		for error in errors:
			logging.error(error)

	def run(self):
		logging.info('Run process')
		self.kill_sock()
		self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self.server.bind(self.sock_path)
		self.server.listen(1)
		logging.info('Listening...')
		print("STATUS: ready", flush=True)
		while True:
			conn, _ = self.server.accept()
			threading.Thread(target=self.handle_session, args=(conn,), daemon=True).start()
		""" 	data = conn.recv(1024).decode()
			for line in data.splitlines():
				response = self.handle_command(line, conn)
			conn.sendall( (response + "\n").encode() )
			if (self.verbose):
				logging.info(response) """

	def run_local(self):
		# Act as client
		logging.info("Running in local mode")
		data = self.init_cmd
		for line in data.splitlines():
			response = self.handle_command(line, None)
		if (self.verbose):
			logging.info(response)
		logging.info("Done.")
		while True:
			user_input = input(">>> ").strip()
			if user_input == "exit" or user_input == "quit":
				break
			elif user_input:
				response = self.handle_command(user_input, None)
				if (self.verbose):
					logging.info(response)
	
	def kill_sock(self):
		if self.server and self.server is not None:
			try:
				self.server.shutdown(socket.SHUT_RDWR)
			except OSError:
				pass  # Already closed or not connected

			self.server.close()
			logging.info("Server socket closed")
		if os.path.exists(args.sock):
			os.unlink(args.sock)

# Set up command-line argument parsing
def parse_args():
	parser = argparse.ArgumentParser(description='GNU Radio Host for SensorGnome')
	parser.add_argument('-s', '--sock', required=True, help='Path to socket', type = str)
	parser.add_argument('-v', '--verbose', default=False, help='Print occurrences in stdout', action="store_true")
	parser.add_argument('-l', '--local', default=False, help='Run with local client socket', action="store_true")
	parser.add_argument('-c', '--cmd', default=None, help='Command to send on init', type = str)
	return parser.parse_args()


if __name__ == "__main__":
	exit_status = 0
	try:
		args = parse_args()
		host = GRH(args.sock, args.verbose, args.cmd)
		if args.local:
			host.run_local()
		else:
			host.run()
	except KeyboardInterrupt:
		logging.error('Process interrupted by user')
		exit_status = 2
	except Exception:
		logging.error(traceback.format_exc())
		exit_status = 1
	finally:
		try:
			host.kill_sock()
		except NameError:
			pass  # host was never created
	exit(exit_status)


